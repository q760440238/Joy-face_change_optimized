#!/usr/bin/env python3
"""16 GB budget attempt: attention int8 + MLP int4 with tiny groups (gs=16)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("/home/ubuntu/jve-diff")
OUT = SRC / "int8_mlp4g16"
OUT.mkdir(exist_ok=True)
GROUP = 16          # was 64 — 4x finer, same 4-bit storage
MIN_NUMEL = 8_000_000


def map_key(k: str) -> str:
    if k.startswith("double_blocks."):
        p = k.split(".")
        if len(p) > 2 and p[2] == "attn":
            return ".".join(p[:2] + p[3:])
    return k


def q8(w):
    wf = w.float()
    sc = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    return (torch.round(wf / sc).clamp_(-127, 127).to(torch.int8).contiguous(),
            sc.to(torch.float16).contiguous())


def q4(w, group=GROUP):
    o, i = w.shape
    wr = w.float().reshape(o, i // group, group)
    sc = wr.abs().amax(dim=-1).clamp_min(1e-8) / 7.0
    q = torch.round(wr / sc.unsqueeze(-1)).clamp_(-8, 7).to(torch.int8)
    q8_ = (q.reshape(o, i) + 8).to(torch.uint8)
    return (q8_[:, 0::2] | (q8_[:, 1::2] << 4)).contiguous(), sc.to(torch.bfloat16).contiguous()


def main():
    t0 = time.time()
    idx = json.load(open(SRC / "transformer_diffusion_pytorch_model.safetensors.index.json"))
    per = {}
    for k, s in idx["weight_map"].items():
        per.setdefault(s, []).append(k)
    out, meta = {}, {}
    errs = {}
    for shard in sorted(per):
        with safe_open(str(SRC / "transformer" / shard), framework="pt") as f:
            for k in per[shard]:
                t = f.get_tensor(k).to(torch.bfloat16)
                nk = map_key(k)
                if t.dim() != 2 or t.numel() < MIN_NUMEL:
                    out[nk] = t.contiguous(); meta[nk] = {"dtype": "bf16"}
                    continue
                if "mlp.net" in nk:
                    q, sc = q4(t)
                    out[nk + ".qweight"] = q; out[nk + ".scales"] = sc
                    meta[nk] = {"dtype": "int4", "group": GROUP}
                    dd = (q & 0x0F).to(torch.float32).sub_(8)
                    deq = torch.empty(t.shape, dtype=torch.float32)
                    deq[:, 0::2] = (q & 0x0F).float() - 8
                    deq[:, 1::2] = (q >> 4).float() - 8
                    deq = (deq.reshape(t.shape[0], -1, GROUP) * sc.float().unsqueeze(-1)).reshape(t.shape)
                    errs.setdefault("int4", []).append(((deq - t.float()).pow(2).mean() / t.float().pow(2).mean()).item())
                else:
                    q, sc = q8(t)
                    out[nk + ".q8weight"] = q; out[nk + ".q8scales"] = sc
                    meta[nk] = {"dtype": "int8"}
                    errs.setdefault("int8", []).append((((q.float() * sc.float()) - t.float()).pow(2).mean() / t.float().pow(2).mean()).item())
    keys = sorted(out)
    third = len(keys) // 3
    for pi, ks in enumerate((keys[:third], keys[third:2*third], keys[2*third:]), 1):
        save_file({k: out[k] for k in ks}, str(OUT / f"q-{pi:05d}-of-00003.safetensors"))
    json.dump({"group_size": GROUP, "tensors": meta}, open(OUT / "meta.json", "w"))
    tot = sum(p.numel() * p.element_size() for p in out.values())
    print(f"size {tot/2**30:.2f} GiB  int4(mlp) rel_mse={sum(errs['int4'])/len(errs['int4']):.5f} "
          f"int8(attn) rel_mse={sum(errs['int8'])/len(errs['int8']):.6f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
