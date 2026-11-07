"""Group-wise int4 weight-only Linear (pure PyTorch -> works on CUDA and ROCm).

Weight storage: packed uint8 `[out, in/2]` (two 4-bit values per byte, low nibble
first) plus bf16 `scales` `[out, in/group]`. Dequantisation happens on the fly in
`forward`; there is no fused kernel, so it runs anywhere torch does.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


class Int4Linear(nn.Module):
    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor,
                 bias: torch.Tensor | None = None, group: int = 64,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.group = int(group)
        self.out_dtype = dtype

    def dequant(self) -> torch.Tensor:
        q = self.qweight
        out_f, half = q.shape
        lo = (q & 0x0F).to(torch.int16) - 8
        hi = (q >> 4).to(torch.int16) - 8
        w = torch.empty((out_f, half * 2), dtype=torch.float32, device=q.device)
        w[:, 0::2] = lo
        w[:, 1::2] = hi
        w = w.view(out_f, -1, self.group) * self.scales.float().unsqueeze(-1)
        return w.reshape(out_f, -1).to(self.out_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequant(), self.bias)

    @torch.no_grad()
    def from_linear_quantized(self, weight: torch.Tensor) -> None:
        """In-place RTN round-trip (for sanity checks against the offline export)."""
        raise NotImplementedError


def _iter_int4_tensors(directory: str):
    d = Path(directory)
    meta = json.load(open(d / "meta.json"))
    group = meta.get("group_size", 64)
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as h:
            for k in h.keys():
                yield f.name, k, h.get_tensor(k), group


def load_int4_state_dict(directory: str, dtype: torch.dtype = torch.bfloat16,
                         mlp_bf16_dir: str | None = None) -> Dict[str, torch.Tensor]:
    """Materialise a bf16 state_dict from the int4 export (dequantised).

    Used for the quality A/B: numerically identical to `Int4Linear.forward`,
    without needing to re-architect the loader.
    """
    packed: Dict[str, torch.Tensor] = {}
    scales: Dict[str, torch.Tensor] = {}
    q8: Dict[str, torch.Tensor] = {}
    s8: Dict[str, torch.Tensor] = {}
    plain: Dict[str, torch.Tensor] = {}
    group = 64
    for _, k, t, group in _iter_int4_tensors(directory):
        if k.endswith(".qweight"):
            packed[k[: -len(".qweight")]] = t
        elif k.endswith(".scales"):
            scales[k[: -len(".scales")]] = t
        elif k.endswith(".q8weight"):
            q8[k[: -len(".q8weight")]] = t
        elif k.endswith(".q8scales"):
            s8[k[: -len(".q8scales")]] = t
        else:
            plain[k] = t
    out: Dict[str, torch.Tensor] = dict(plain)
    for base, q in packed.items():
        s = scales[base]
        out_f, half = q.shape
        lo = (q & 0x0F).to(torch.int16) - 8
        hi = (q >> 4).to(torch.int16) - 8
        w = torch.empty((out_f, half * 2), dtype=torch.float32)
        w[:, 0::2] = lo.to(torch.float32)
        w[:, 1::2] = hi.to(torch.float32)
        w = w.view(out_f, -1, group) * s.float().unsqueeze(-1)
        out[base] = w.reshape(out_f, -1).to(dtype)
    for base, qw in q8.items():
        out[base] = (qw.float() * s8[base].float()).to(dtype)
    if mlp_bf16_dir:
        import glob as _glob
        n = 0
        for f in sorted(_glob.glob(str(Path(mlp_bf16_dir) / "*.safetensors"))):
            with safe_open(f, framework="pt", device="cpu") as h:
                for k in h.keys():
                    if "mlp" in k:
                        out[k] = h.get_tensor(k).to(dtype)
                        n += 1
        print(f"[int4] kept {n} MLP tensors in bf16 from {mlp_bf16_dir}", flush=True)
    return out


def _map_key(k: str) -> str:
    if k.startswith("double_blocks."):
        parts = k.split(".")
        if len(parts) > 2 and parts[2] == "attn":
            return ".".join(parts[:2] + parts[3:])
    return k


def load_safetensors_state_dict(directory: str, dtype=torch.bfloat16):
    """Read the official diffusers shards as a runtime-named state_dict."""
    d = Path(directory)
    files = sorted(d.glob("*.safetensors")) or sorted((d / "transformer").glob("*.safetensors"))
    out = {}
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as h:
            for k in h.keys():
                out[_map_key(k)] = h.get_tensor(k).to(dtype)
    return out
