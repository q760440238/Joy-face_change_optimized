#!/usr/bin/env python3
"""Numerical equivalence test: joyomni_ops CUDA kernels vs the torch fallback.

    python test_fused_equiv.py

Reports max abs / max rel error per op on the shapes the 840x480 pipeline hits.
"""
from __future__ import annotations

import torch

import joyomni_ops
from xvideo.models.dit import fused_ops_torch as T


def report(name, a, b):
    d = (a.float() - b.float()).abs()
    denom = b.float().abs().clamp_min(1e-6)
    print(
        f"{name:<26} max_abs={d.max().item():.3e}  "
        f"max_rel={(d / denom).max().item():.3e}  "
        f"mean_abs={d.mean().item():.3e}  exact={(a == b).float().mean().item()*100:.1f}%"
    )


def main():
    torch.manual_seed(0)
    dev = "cuda"
    dtype = torch.bfloat16

    # --- fused_norm_scale_shift (LayerNorm + adaLN modulate) ---------------
    M, N = 4096, 3072
    x = torch.randn(M, N, device=dev, dtype=dtype)
    gamma = torch.randn(N, device=dev, dtype=dtype)
    beta = torch.randn(N, device=dev, dtype=dtype)
    scale = torch.randn(M, N, device=dev, dtype=dtype) * 0.1
    shift = torch.randn(M, N, device=dev, dtype=dtype) * 0.1
    a = joyomni_ops.fused_norm_scale_shift(x, gamma, beta, scale, shift, "layer", 1e-6)
    b = T.fused_norm_scale_shift(x, gamma, beta, scale, shift, "layer", 1e-6)
    report("norm_scale_shift layer", a, b)

    a = joyomni_ops.fused_norm_scale_shift(x, gamma, None, scale, shift, "rms", 1e-6)
    b = T.fused_norm_scale_shift(x, gamma, None, scale, shift, "rms", 1e-6)
    report("norm_scale_shift rms", a, b)

    # --- rmsnorm ----------------------------------------------------------
    Mr, Nr = 8192, 128
    xr = torch.randn(Mr, Nr, device=dev, dtype=dtype)
    wr = torch.randn(Nr, device=dev, dtype=dtype)
    report("rmsnorm", joyomni_ops.rmsnorm(xr, wr, 1e-6), T.rmsnorm(xr, wr, 1e-6))

    # --- fused_qk_norm_rope_3d_paired ------------------------------------
    B, L, H, D = 1, 256, 24, 128
    q = torch.randn(B, L * H, D, device=dev, dtype=dtype)
    k = torch.randn(B, L * H, D, device=dev, dtype=dtype)
    qw = torch.randn(D, device=dev, dtype=dtype) * 0.5 + 1.0
    kw = torch.randn(D, device=dev, dtype=dtype) * 0.5 + 1.0
    ang = torch.rand(L, D // 2, device=dev) * 6.283
    cos = torch.cos(ang).to(dtype)
    sin = torch.sin(ang).to(dtype)

    q1, k1 = q.clone(), k.clone()
    joyomni_ops.fused_qk_norm_rope_3d_paired(q1, k1, L, H, 1e-6, qw, kw, cos, sin)
    q2, k2 = q.clone(), k.clone()
    T.fused_qk_norm_rope_3d_paired(q2, k2, L, H, 1e-6, qw, kw, cos, sin)
    report("qk_norm_rope  q", q1, q2)
    report("qk_norm_rope  k", k1, k2)

    # --- fused_add_gate (already pure torch, sanity only) -----------------
    res = torch.randn(B, L, D, device=dev, dtype=dtype)
    xx = torch.randn(B, L, D, device=dev, dtype=dtype)
    g = torch.rand(B, 1, device=dev, dtype=dtype)
    report("add_gate", torch.addcmul(res, xx, g.unsqueeze(1)),
           torch.addcmul(res, xx, g.unsqueeze(1)))


if __name__ == "__main__":
    main()
