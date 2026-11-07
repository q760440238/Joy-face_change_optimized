#!/usr/bin/env python3
"""Cost of the torch fallback vs the CUDA kernels (same shapes, warm loop)."""
import time

import torch

import joyomni_ops
from xvideo.models.dit import fused_ops_torch as T


def bench(fn, iters=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


torch.manual_seed(0)
dev, dt = "cuda", torch.bfloat16

M, N = 4096, 3072
x = torch.randn(M, N, device=dev, dtype=dt)
g = torch.randn(N, device=dev, dtype=dt)
b = torch.randn(N, device=dev, dtype=dt)
sc = torch.randn(M, N, device=dev, dtype=dt) * 0.1
sh = torch.randn(M, N, device=dev, dtype=dt) * 0.1
t_cuda = bench(lambda: joyomni_ops.fused_norm_scale_shift(x, g, b, sc, sh, "layer", 1e-6))
t_torch = bench(lambda: T.fused_norm_scale_shift(x, g, b, sc, sh, "layer", 1e-6))
print(f"layernorm_modulate  cuda={t_cuda:7.3f} ms  torch={t_torch:7.3f} ms  x{t_torch / t_cuda:.2f}")

L, H, D = 256, 24, 128
q = torch.randn(1, L * H, D, device=dev, dtype=dt)
k = torch.randn(1, L * H, D, device=dev, dtype=dt)
qw = torch.randn(D, device=dev, dtype=dt)
kw = torch.randn(D, device=dev, dtype=dt)
ang = torch.rand(L, D // 2, device=dev) * 6.283
cos = torch.cos(ang).to(dt)
sin = torch.sin(ang).to(dt)
t_cuda = bench(lambda: joyomni_ops.fused_qk_norm_rope_3d_paired(q, k, L, H, 1e-6, qw, kw, cos, sin))
t_torch = bench(lambda: T.fused_qk_norm_rope_3d_paired(q, k, L, H, 1e-6, qw, kw, cos, sin))
print(f"qk_norm_rope        cuda={t_cuda:7.3f} ms  torch={t_torch:7.3f} ms  x{t_torch / t_cuda:.2f}")
