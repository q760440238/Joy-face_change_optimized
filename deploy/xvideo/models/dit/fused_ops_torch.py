"""Pure-PyTorch equivalents of the joyomni_ops CUDA kernels.

Mirrors `deploy/joyomni_ops/csrc/{fused_norm_scale_shift,fused_qknorm_rope_3d_kernel,rmsnorm}.cu`
so the DiT can run on any PyTorch backend (ROCm, CPU, MPS) without the CUDA
extension. All maths is done in fp32 and rounded back to the input dtype,
matching the kernels' `float` accumulators.

Selected automatically when `joyomni_ops` cannot be imported, or forced with
`JOYOMNI_FUSED_TORCH=1`.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

_TRUTHY = {"1", "true", "yes", "on"}


def forced() -> bool:
    return os.environ.get("JOYOMNI_FUSED_TORCH", "").strip().lower() in _TRUTHY


def _norm_scale_shift(
    x_2d: torch.Tensor,
    gamma: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    scale_2d: torch.Tensor,
    shift_2d: torch.Tensor,
    norm_type: str,
    eps: float,
) -> torch.Tensor:
    """out = Norm(x; gamma, beta) * (1 + scale) + shift, fp32 math."""
    xf = x_2d.float()
    if norm_type == "layer":
        mean = xf.mean(dim=-1, keepdim=True)
        var = xf.var(dim=-1, unbiased=False, keepdim=True)
        y = (xf - mean) * torch.rsqrt(var + eps)
        if gamma is not None:
            y = y * gamma.float().view(1, -1)
        if beta is not None:
            y = y + beta.float().view(1, -1)
    else:  # rms
        y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        if gamma is not None:
            y = y * gamma.float().view(1, -1)
    y = y * (1.0 + scale_2d.float()) + shift_2d.float()
    return y.to(x_2d.dtype)


def fused_norm_scale_shift(
    x: torch.Tensor,
    gamma: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_type: str = "layer",
    eps: float = 1e-6,
) -> torch.Tensor:
    return _norm_scale_shift(x, gamma, beta, scale, shift, norm_type, eps)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = y * weight.float().view(1, -1)
    return y.to(x.dtype)


def fused_qk_norm_rope_3d_paired(
    q: torch.Tensor,          # [B, L*H, D] bf16, updated in place
    k: torch.Tensor,          # [B, L*H, D] bf16, updated in place
    seq_len: int,
    num_heads: int,
    eps: float,
    q_weight: torch.Tensor,   # [D]
    k_weight: torch.Tensor,   # [D]
    cos: torch.Tensor,        # [L, D/2] bf16
    sin: torch.Tensor,        # [L, D/2] bf16
) -> None:
    B, LH, D = q.shape
    L, H = seq_len, num_heads
    assert LH == L * H, (LH, L, H)

    def _norm_rope(t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        tf = t.view(B, L, H, D).float()
        tf = tf * torch.rsqrt(tf.pow(2).mean(dim=-1, keepdim=True) + eps)
        tf = tf * w.float().view(1, 1, 1, D)
        c = cos.float().view(1, L, 1, D // 2)
        s = sin.float().view(1, L, 1, D // 2)
        even = tf[..., 0::2]
        odd = tf[..., 1::2]
        out = torch.empty_like(tf)
        out[..., 0::2] = even * c - odd * s
        out[..., 1::2] = even * s + odd * c
        return out.reshape(B, LH, D).to(t.dtype)

    q.copy_(_norm_rope(q, q_weight))
    k.copy_(_norm_rope(k, k_weight))
