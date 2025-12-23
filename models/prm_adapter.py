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


class PRMAdapter(nn.Module):
    """
    Phase-aware Reliability Modulation (PRM) Adapter for DFER.

    Args:
        dim: token embedding dim D
        inner_dim: hidden dim for Mona path
        tau: temperature for reliability gate
    """
    def __init__(self, dim: int, inner_dim: int, tau: float = 1.0):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim
        self.tau = float(tau)

        self.ln = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(dim))

        self.proj_in = nn.Linear(dim, inner_dim)
        self.mona = MonaOp(inner_dim)
        self.proj_out = nn.Linear(inner_dim, dim)

        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, u: torch.Tensor, B: int, T: int, hw: Tuple[int, int]):
        """
        Args:
            u: [L+1, B*T, D] or [B*T, L+1, D] attention output
            B: batch size
            T: temporal length
            hw: (H, W) spatial resolution of patch tokens
        Returns:
            residual: gated residual to add to x, same shape as u
            r: frame-wise variation r_t, [B, T]
            beta: reliability gate beta_t, [B, T]
        """
        H, W = hw

        # Normalize to [B*T, L, D]
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
        if L != 1 + H * W:
            raise ValueError(f"L should be 1+H*W, got L={L}, H*W={H*W}")
        if D != self.dim:
            raise ValueError(f"Channel dim mismatch: {D} vs {self.dim}")

        # Calibration with scaled LN
        u_cal = self.ln(u_btld) * self.gamma + u_btld * self.gammax  # [B*T, L, D]

        # Temporal variation from CLS token
        cls = u_cal[:, 0, :]              # [B*T, D]
        cls = cls.view(B, T, D)           # [B, T, D]
        r_delta = cls[:, 1:] - cls[:, :-1]  # [B, T-1, D]
        r = torch.norm(r_delta, dim=-1)      # [B, T-1]
        r = F.pad(r, (1, 0), value=0.0)      # [B, T], first frame variation set to 0

        beta = torch.exp(-r / max(self.tau, 1e-6)).clamp(min=0.0, max=1.0)  # [B, T]
        beta_bt = beta.view(BT, 1, 1)                                       # [B*T,1,1]

        # Mona residual on patches (skip CLS)
        z = self.proj_in(u_cal)                              # [B*T, L, C]
        z_cls = z[:, :1, :]                                  # [B*T,1,C]
        z_patch = z[:, 1:, :]                                # [B*T,H*W,C]

        z_patch_2d = z_patch.view(BT, H, W, -1).permute(0, 3, 1, 2)  # [B*T,C,H,W]
        z_patch_2d = self.mona(z_patch_2d)                          # [B*T,C,H,W]
        z_patch = z_patch_2d.permute(0, 2, 3, 1).contiguous().view(BT, H * W, -1)

        z = torch.cat([torch.zeros_like(z_cls), z_patch], dim=1)     # CLS kept zeroed
        delta_feat = self.proj_out(z)                                # [B*T, L, D]

        residual = self.alpha * beta_bt * delta_feat                 # [B*T, L, D]

        if transposed:
            residual = residual.permute(1, 0, 2).contiguous()        # [L, B*T, D]

        return residual, r, beta
