#!/usr/bin/env python3
"""All-int8 (per-output-channel symmetric) quantization of the DiT weights."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("/home/ubuntu/jve-diff")
OUT = SRC / "int8"
OUT.mkdir(exist_ok=True)
MIN_NUMEL = 8_000_000


def map_key(k: str) -> str:
    if k.startswith("double_blocks."):
        p = k.split(".")
        if len(p) > 2 and p[2] == "attn":
            return ".".join(p[:2] + p[3:])
    return k


def q8(w: torch.Tensor):
    wf = w.float()
    scale = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.round(wf / scale).clamp_(-127, 127).to(torch.int8).contiguous()
    return q, scale.to(torch.float16).contiguous()


def main():
    t0 = time.time()
    idx = json.load(open(SRC / "transformer_diffusion_pytorch_model.safetensors.index.json"))
    per_shard = {}
    for k, s in idx["weight_map"].items():
        per_shard.setdefault(s, []).append(k)
    out_t, meta = {}, {}
    n_q = 0
    errs = []
    for shard in sorted(per_shard):
        with safe_open(str(SRC / "transformer" / shard), framework="pt") as f:
            for k in per_shard[shard]:
                t = f.get_tensor(k).to(torch.bfloat16)
                nk = map_key(k)
                if t.dim() == 2 and t.numel() >= MIN_NUMEL:
                    q, sc = q8(t)
                    out_t[nk + ".q8weight"] = q
                    out_t[nk + ".q8scales"] = sc
                    meta[nk] = {"dtype": "int8", "shape": list(t.shape)}
                    n_q += 1
                    if n_q % 40 == 0:
                        deq = q.float() * sc.float()
                        errs.append(((deq - t.float()).pow(2).mean() / t.float().pow(2).mean()).item())
                else:
                    out_t[nk] = t.contiguous()
                    meta[nk] = {"dtype": "bf16", "shape": list(t.shape)}
    keys = sorted(out_t)
    third = len(keys) // 3
    parts = [keys[:third], keys[third:2 * third], keys[2 * third:]]
    for pi, ks in enumerate(parts, 1):
        save_file({k: out_t[k] for k in ks}, str(OUT / f"int8-{pi:05d}-of-0000{len(parts)}.safetensors"))
    json.dump({"group_size": 0, "tensors": meta}, open(OUT / "meta.json", "w"))
    tot = sum(p.numel() * p.element_size() for p in out_t.values())
    print(f"quantized {n_q} tensors, size {tot/2**30:.2f} GiB, "
          f"mean rel_mse {sum(errs)/len(errs):.5f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
