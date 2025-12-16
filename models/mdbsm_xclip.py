from typing import Tuple, Union
import torch
from torch import nn
import numpy as np
from mit import MultiframeIntegrationTransformer
from prompt import VideoSpecificPrompt
from cct import CrossFrameCommunicationTransformer
import sys
import warnings
sys.path.append("../")
from clip.model import CLIP,LayerNorm,Transformer,QuickGELU
import clip


class ModalDecompositionModule(nn.Module):
    """Extract dominant expression mode and distill frame tokens toward it."""

    def __init__(self, dim: int, num_heads: int = 8, alpha_init: float = 0.1):
        super().__init__()
        self.modal_query = nn.Parameter(torch.randn(dim))  # 可学习模态查询，抓取全局主表达模式
        self.modal_attn = nn.MultiheadAttention(dim, num_heads)  # 轻量跨帧注意力
        self.align_mlp = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim * 2),
            QuickGELU(),
            nn.Linear(dim * 2, dim),
        )
        self.alpha = nn.Parameter(torch.tensor(alpha_init))  # 对齐强度

    def forward(self, features: torch.Tensor):
        # features: [B, T, D]
        b, t, d = features.shape
        modal_query = self.modal_query.to(features).unsqueeze(0).expand(b, -1)  # [B, D]
        modal_query = modal_query.unsqueeze(0)  # [1, B, D] for MultiheadAttention

        # dominant expression mode
        modal = self.modal_attn(modal_query, features.permute(1, 0, 2), features.permute(1, 0, 2), need_weights=False)[0]
        modal = modal.squeeze(0)  # [B, D]

        # static modal distillation: 将每帧向主模态柔性投影
        modal_expand = modal.unsqueeze(1).expand(-1, t, -1)
        align_input = torch.cat([features, modal_expand], dim=-1)
        distilled = features + self.alpha * self.align_mlp(align_input)
        return modal, distilled


class BoilingSuppressionModule(nn.Module):
    """Detect modal boiling and softly reflow features toward the dominant mode."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, distilled: torch.Tensor, modal: torch.Tensor, text_features: torch.Tensor, logit_scale: torch.Tensor):
        # distilled: [B, T, D], modal: [B, D], text_features: [B, K, D]
        b, t, d = distilled.shape
        modal_expand = modal.unsqueeze(1).expand(-1, t, -1)

        deviation = (distilled - modal_expand).norm(dim=-1, keepdim=True)  # [B, T, 1] 偏离主模态的幅度

        # frame-level uncertainty
        frame_feats = distilled / (distilled.norm(dim=-1, keepdim=True) + 1e-6)
        norm_text = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
        logits = torch.einsum("btd,bkd->btk", frame_feats, norm_text) * logit_scale
        probs = logits.softmax(dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)  # [B, T, 1]

        delta_entropy = torch.zeros_like(entropy)
        delta_entropy[:, 1:] = entropy[:, 1:] - entropy[:, :-1]
        delta2_entropy = torch.zeros_like(entropy)
        delta2_entropy[:, 1:] = delta_entropy[:, 1:] - delta_entropy[:, :-1]

        # 组合偏差与熵一阶/二阶差分，构成沸腾检测描述子
        descriptor = torch.cat([deviation, entropy, delta_entropy, delta2_entropy], dim=-1)  # [B, T, 4]
        gamma = self.gate(descriptor)  # [B, T, 1]
        stabilized = gamma * distilled + (1.0 - gamma) * modal_expand  # 自适应回流到主模态

        return stabilized, logits, probs, gamma.squeeze(-1)


class ModalSpectrumAlignment(nn.Module):
    """Regularize spectrum compactness over the stabilized trajectory."""

    def __init__(self, preserve_k: int = 4, weight: float = 1e-2):
        super().__init__()
        self.preserve_k = preserve_k
        self.weight = weight

    def forward(self, stabilized: torch.Tensor):
        if self.weight <= 0:
            return stabilized.new_tensor(0.0)

        b, t, d = stabilized.shape
        centered = stabilized - stabilized.mean(dim=1, keepdim=True)
        cov = torch.matmul(centered.transpose(1, 2), centered) / float(t)  # [B, D, D] 序列协方差
        eigvals = torch.linalg.eigvalsh(cov.float())  # ascending
        eigvals = eigvals.to(stabilized)
        eigvals = torch.flip(eigvals, dims=[1])  # descending

        tail = eigvals[:, self.preserve_k:]
        spec_loss = (tail.sum(dim=1) / float(d)).mean()
        return spec_loss * self.weight

class XCLIP(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int, 
                 # video
                 T=8, 
                 droppath=0.,
                 mit_layers=1,
                 # prompt 
                 prompts_alpha=1e-4,
                 prompts_layers=1,
                 # stabilization
                 mdm_heads=8,
                 mdm_alpha=0.1,
                 bsm_hidden=64,
                 msa_preserve_k=4,
                 msa_weight=1e-2,
                 # other
                 use_cache=True,
                 use_checkpoint=False,
                 ):
        super().__init__(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
        
        self.prompts_generator = VideoSpecificPrompt(layers=prompts_layers, embed_dim=embed_dim, alpha=prompts_alpha,)
        self.use_cache=use_cache
        self.mit = MultiframeIntegrationTransformer(T=T, embed_dim=embed_dim, layers=mit_layers,)

        dpr = [x.item() for x in torch.linspace(0, droppath, vision_layers)] if droppath > 0. else None

        vision_heads = vision_width // 64
        self.visual = CrossFrameCommunicationTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            droppath=dpr,
            T=T,
            use_checkpoint=use_checkpoint,
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.cache_text_features = None
        self.prompts_visual_ln = LayerNorm(vision_width)
        self.prompts_visual_proj = nn.Parameter(torch.randn(vision_width, embed_dim))

        # MDBSM components
        self.mdm = ModalDecompositionModule(dim=embed_dim, num_heads=mdm_heads, alpha_init=mdm_alpha)
        self.bsm = BoilingSuppressionModule(hidden_dim=bsm_hidden)
        self.msa = ModalSpectrumAlignment(preserve_k=msa_preserve_k, weight=msa_weight)
        
        self.initialize_parameters()
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'positional_embedding'}

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)
        eos_indx = text.argmax(dim=-1)
        K, N1, C = x.shape

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(K, -1)
        return x

    def encode_video(self, image, text_features: torch.Tensor, logit_scale: torch.Tensor):
        b,t,c,h,w = image.size()
        image = image.reshape(-1,c,h,w)

        cls_features, img_features = self.encode_image(image)
        img_features = self.prompts_visual_ln(img_features)
        img_features = img_features @ self.prompts_visual_proj
        
        cls_features = cls_features.view(b, t, -1)
        img_features = img_features.view(b, t, -1, cls_features.shape[-1])

        text_features = text_features.to(cls_features)

        # MDM：主模态提取与静态蒸馏
        modal, distilled = self.mdm(cls_features)
        # BSM：利用偏差与熵动态检测沸腾并柔性回流
        stabilized, frame_logits, frame_probs, gates = self.bsm(distilled, modal, text_features, logit_scale)
        # MSA：序列谱压缩正则
        spec_loss = self.msa(stabilized)
        # 跨帧整合得到视频级表示
        video_features = self.mit(stabilized)

        return {
            "video_features": video_features,
            "frame_features": stabilized,
            "dominant_mode": modal,
            "frame_logits": frame_logits,
            "frame_probs": frame_probs,
            "gates": gates,
            "spec_loss": spec_loss,
            "img_features": img_features,
        }

    def cache_text(self, text):
        self.eval()
        with torch.no_grad():
            if self.cache_text_features is None:
                self.cache_text_features = self.encode_text(text)
        self.train()
        return self.cache_text_features

    def forward(self, image, text, return_features: bool = True):
        b = image.shape[0]

        if self.use_cache:
            text_features = self.cache_text(text)
        else:
            text_features = self.encode_text(text)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
        text_features = text_features.unsqueeze(0).expand(b, -1, -1)

        logit_scale = self.logit_scale.exp()
        video_outputs = self.encode_video(image, text_features, logit_scale)

        video_features = video_outputs["video_features"]
        video_features = video_features / (video_features.norm(dim=-1, keepdim=True) + 1e-6)
        logits = torch.einsum("bd,bkd->bk", video_features, logit_scale * text_features.to(video_features))

        aux = {
            "dominant_mode": video_outputs["dominant_mode"],
            "frame_features": video_outputs["frame_features"],
            "frame_logits": video_outputs["frame_logits"],
            "frame_probs": video_outputs["frame_probs"],
            "gates": video_outputs["gates"],
            "img_features": video_outputs["img_features"],
        }

        if return_features:
            return logits, video_outputs["spec_loss"], aux
        return logits


def build_model(state_dict: dict, T=8, droppath=0., use_checkpoint=False, logger=None, prompts_alpha=1e-1, prompts_layers=2, use_cache=True, mit_layers=4,):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    model = XCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,  
        T=T, droppath=droppath, mit_layers=mit_layers,
        prompts_alpha=prompts_alpha, prompts_layers=prompts_layers,
        use_checkpoint=use_checkpoint, use_cache=use_cache,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    msg = model.load_state_dict(state_dict,strict=False)
    logger.info(f"load pretrained CLIP: {msg}")
    
    return model.eval()


def load(model_path, name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", 
         jit=True, T=8, droppath=0., use_checkpoint=False, logger=None, use_cache=True, prompts_alpha=1e-1, prompts_layers=2, mit_layers=1,
):
    if model_path is None:
        model_path = clip._download(clip._MODELS[name])
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(state_dict or model.state_dict(), T=T, droppath=droppath, 
                        use_checkpoint=use_checkpoint, logger=logger,
                        prompts_alpha=prompts_alpha, 
                        prompts_layers=prompts_layers,
                        use_cache=use_cache,
                        mit_layers=mit_layers,
                        )
    if str(device) == "cpu":
        model.float()
    return model, model.state_dict()
