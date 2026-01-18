import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MonaOp(nn.Module):
    """
    Mona Operator: Multi-scale depth-wise convolution.
    Input/Output: [B, C, H, W]
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3//2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5//2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7//2, groups=in_features)
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
        x = self.projector(x)
        return x



class MCPAdapter(nn.Module):
    """
    Multi-Cognitive Perceptual (MCP) adapter for spatial enhancement.
    Works on per-frame patch tokens with a lightweight temporal branch.
    Expects patch tokens only (no CLS).
    """
    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim

        self.ln = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, inner_dim)
        self.mona = MonaOp(inner_dim)
        self.proj_out = nn.Linear(inner_dim, dim)
        # 1x1 卷积分支用于时间建模（按 token 的时间序列）。
        self.temporal_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.temporal_proj = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, u: torch.Tensor, B: int, T: int, hw: Tuple[int, int]):
        """
        Args:
            u: [L+1, B*T, D] or [B*T, L+1, D] attention output
            B: batch size
            T: temporal length
            hw: (H, W) spatial resolution of patch tokens
        Returns:
            residual: spatial residual to add to u, same shape as u
        """
        H, W = hw
        transposed = False
        if u.dim() != 3:
            raise ValueError(f"u must be 3D, got {u.shape}")
        if u.shape[0] == B * T:
            u_btld = u
        elif u.shape[1] == B * T:
            u_btld = u.permute(1, 0, 2).contiguous()
            transposed = True
        else:
            raise ValueError(f"Unexpected u shape {u.shape} for B={B}, T={T}")

        BT, L, D = u_btld.shape
        if BT != B * T:
            raise ValueError(f"BT mismatch: {BT} vs {B*T}")
        if L != H * W:
            raise ValueError(f"L should be H*W, got L={L}, H*W={H*W}")
        if D != self.dim:
            raise ValueError(f"Channel dim mismatch: {D} vs {self.dim}")

        # 逐帧空间增强，不引入跨帧注意力。
        u_cal = self.ln(u_btld)
        z = self.proj_in(u_cal)
        z_patch = z
        z_patch_2d = z_patch.view(BT, H, W, -1).permute(0, 3, 1, 2)
        # 多尺度深度可分离卷积聚合空间线索。
        z_patch_2d = self.mona(z_patch_2d)
        z_patch = z_patch_2d.permute(0, 2, 3, 1).contiguous().view(BT, H * W, -1)
        delta_spatial = self.proj_out(z_patch)

        # 时间建模分支：对 patch tokens 的时间序列做 1x1 卷积。
        u_time = u_cal.view(B, T, L, D)
        u_time_patch = u_time                                                # [B,T,L,D]
        u_time_patch = u_time_patch.permute(0, 2, 1, 3).contiguous()         # [B,L-1,T,D]
        u_time_patch = u_time_patch.view(B * L, T, D).permute(0, 2, 1)       # [B*L,D,T]
        u_time_patch = self.temporal_conv(u_time_patch).permute(0, 2, 1)     # [B*(L-1),T,D]
        u_time_patch = u_time_patch.view(B, L, T, D).permute(0, 2, 1, 3).contiguous()      # [B,T,L,D]
        u_time_patch = u_time_patch.view(B * T, L, D)
        delta_time_patch = self.temporal_proj(u_time_patch)
        delta_time = delta_time_patch

        delta_feat = 0.5 * (delta_spatial + delta_time)

        # 残差注入回 token 表示。
        residual = self.alpha * delta_feat
        if transposed:
            residual = residual.permute(1, 0, 2).contiguous()
        return residual
