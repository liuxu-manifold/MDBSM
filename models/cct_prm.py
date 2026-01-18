from collections import OrderedDict
from timm.models.layers import trunc_normal_
import torch
from torch import nn
import os
import sys
from prm_adapter import MCPAdapter
sys.path.append("../")
from clip.model import LayerNorm, QuickGELU, DropPath

_VIDEOMAMBA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "VideoMamba-main"))
if _VIDEOMAMBA_ROOT not in sys.path:
    sys.path.append(_VIDEOMAMBA_ROOT)
_MAMBA_ROOT = os.path.join(_VIDEOMAMBA_ROOT, "mamba")
if _MAMBA_ROOT not in sys.path:
    sys.path.append(_MAMBA_ROOT)
_CAUSAL_CONV1D_ROOT = os.path.join(_VIDEOMAMBA_ROOT, "causal-conv1d")
if _CAUSAL_CONV1D_ROOT not in sys.path:
    sys.path.append(_CAUSAL_CONV1D_ROOT)
from videomamba.image_sm.models.videomamba import create_block




class TemporalReliabilityMamba(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 tau: float = 1.0, gamma: float = 1.0,
                 rms_norm: bool = False, fused_add_norm: bool = False):
        super().__init__()
        self.dim = dim
        self.tau = float(tau)
        self.gamma = float(gamma)

        ssm_cfg = {"d_state": d_state, "d_conv": d_conv, "expand": expand}
        self.block_f = create_block(
            dim,
            ssm_cfg=ssm_cfg,
            bimamba=False,
            # rms_norm=rms_norm,
            # fused_add_norm=fused_add_norm,
        )
        self.block_b = create_block(
            dim,
            ssm_cfg=ssm_cfg,
            bimamba=False,
            # rms_norm=rms_norm,
            # fused_add_norm=fused_add_norm,
        )
        self.ln = nn.LayerNorm(dim)
        self.delta_scale = 0.1

    def _run_block(self, x: torch.Tensor, block: nn.Module):
        hidden, _ = block(x, residual=None, inference_params=None)
        return (hidden - x) * self.delta_scale

    def _compute_beta(self, h_f: torch.Tensor, h_b: torch.Tensor):
        ln_f = self.ln(h_f)
        ln_b = self.ln(h_b)
        cos_sim = torch.nn.functional.cosine_similarity(ln_f, ln_b, dim=-1)
        c_t = 1.0 - cos_sim

        hf_prev = torch.cat([h_f[:, :1], h_f[:, :-1]], dim=1)
        hb_next = torch.cat([h_b[:, 1:], h_b[:, -1:]], dim=1)
        sigma2 = 0.5 * (
            (h_f - hf_prev).pow(2).sum(dim=-1)
            +
            (h_b - hb_next).pow(2).sum(dim=-1)
        )

        log_beta = -(c_t / max(self.tau, 1e-6) + sigma2 / max(self.gamma, 1e-6))
        log_beta = log_beta - log_beta.max(dim=1, keepdim=True)[0]
        beta = torch.exp(log_beta).clamp(min=0.0, max=1.0)
        return beta

    def forward(self, x: torch.Tensor, stop_gradient: bool = True):
        """
        Args:
            x: [B, T, D] frame-level features
            stop_gradient: detach beta from graph if True
        Returns:
            h: [B, T, D] refined temporal states
            beta: [B, T] reliability prior
        """
        # 方向特定的时间动态探针。
        delta_f = self._run_block(x, self.block_f)
        delta_b = self._run_block(torch.flip(x, dims=[1]), self.block_b)
        delta_b = torch.flip(delta_b, dims=[1])

        b, t, d = x.shape
        # 无门控的前后向状态展开用于可靠性估计。
        h_f = torch.zeros_like(x)
        h_b = torch.zeros_like(x)
        h_f[:, 0] = x[:, 0]
        h_b[:, t - 1] = x[:, t - 1]
        for i in range(1, t):
            h_f[:, i] = h_f[:, i - 1] + delta_f[:, i]
        for i in range(t - 2, -1, -1):
            h_b[:, i] = h_b[:, i + 1] + delta_b[:, i]

        beta = self._compute_beta(h_f, h_b)
        if stop_gradient:
            # 可靠性推断与判别学习解耦（stop-gradient）。
            beta = beta.detach()

        # 基于可靠性的门控状态细化。
        h_f_ref = torch.zeros_like(x)
        h_b_ref = torch.zeros_like(x)
        h_f_ref[:, 0] = x[:, 0]
        h_b_ref[:, t - 1] = x[:, t - 1]
        for i in range(1, t):
            gate = beta[:, i].unsqueeze(-1)
            h_f_ref[:, i] = h_f_ref[:, i - 1] + gate * delta_f[:, i]
        for i in range(t - 2, -1, -1):
            gate = beta[:, i].unsqueeze(-1)
            h_b_ref[:, i] = (1.0 - gate) * h_b_ref[:, i + 1] + gate * delta_b[:, i]

        h = 0.5 * (h_f_ref + h_b_ref)
        return h, beta


class CrossFramelAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, droppath = 0., T=0):
        super().__init__()
        self.T = T

        self.message_fc = nn.Linear(d_model, d_model)
        self.message_ln = LayerNorm(d_model)
        self.message_attn = nn.MultiheadAttention(d_model, n_head,)

        self.attn = nn.MultiheadAttention(d_model, n_head,)
        self.ln_1 = LayerNorm(d_model)

        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.message_lstm = nn.LSTM(input_size=1024, hidden_size=1024,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.message_proj = nn.Linear(
            2048, 1024, bias=False)
        self.mcp_adapter = MCPAdapter(dim=d_model, inner_dim=d_model//4)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self, x):
        l, bt, d = x.size()

        if self.T <= 0 or bt % self.T != 0:
            raise ValueError(f"Temporal length mismatch: bt={bt}, T={self.T}")

        attn_out = self.attention(self.ln_1(x))
        x = x + self.drop_path(attn_out)

        hw_tokens = l - 1
        hw_side = int(hw_tokens ** 0.5)
        if hw_side * hw_side != hw_tokens:
            raise ValueError(f"Patch tokens not square: l={l}, hw_tokens={hw_tokens}")
        cls_tok = x[:1]      # [1, BT, D]
        patch_tok = x[1:]    # [HW, BT, D]
        res_patch = self.mcp_adapter(patch_tok, B=bt//self.T, T=self.T, hw=(hw_side, hw_side))
        patch_tok = patch_tok + self.drop_path(res_patch)
        x = torch.cat([cls_tok, patch_tok], dim=0)
        x = x + self.drop_path(self.mlp(self.ln_2(x)))

        return x, None


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, droppath=None,
                 use_checkpoint=False, T=8):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if droppath is None:
            droppath = [0.0 for i in range(layers)]
        self.width = width
        self.layers = layers

        # need ModuleList because blocks return tuple (x, beta)
        self.resblocks = nn.ModuleList([
            CrossFramelAttentionBlock(
                width,
                heads,
                attn_mask,
                droppath[i],
                T,
            )
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        beta = None
        for blk in self.resblocks:
            x, beta = blk(x)
        return x, beta


class CrossFrameCommunicationTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 droppath = None, T = 8, use_checkpoint = False,
                 mamba_d_state: int = 16, mamba_d_conv: int = 4, mamba_expand: int = 2,
                 tau_beta: float = 1.0, gamma_beta: float = 1.0):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.T = T

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        ## Attention Blocks
        self.transformer = Transformer(
            width,
            layers,
            heads,
            droppath=droppath,
            use_checkpoint=use_checkpoint,
            T=T,
        )
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.temporal_reliability = TemporalReliabilityMamba(
            dim=output_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            tau=tau_beta,
            gamma=gamma_beta,
        )


    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x, beta = self.transformer(x)
        x = x.permute(1, 0, 2)

        cls_x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            cls_x = cls_x @ self.proj

        if self.T <= 0 or cls_x.shape[0] % self.T != 0:
            raise ValueError(f"Temporal length mismatch: BT={cls_x.shape[0]}, T={self.T}")
        b = cls_x.shape[0] // self.T
        frame_x = cls_x.view(b, self.T, -1)
        frame_x, beta = self.temporal_reliability(frame_x, stop_gradient=True)
        cls_x = frame_x.reshape(-1, frame_x.shape[-1])

        return cls_x, x[:,1:,:], beta
