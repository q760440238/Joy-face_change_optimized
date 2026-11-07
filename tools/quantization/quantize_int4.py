#!/usr/bin/env python3
"""Group-wise symmetric int4 quantization of the JoyAI-Video-Edit DiT.

Reads the official diffusers shards, renames `double_blocks.N.attn.X` ->
`double_blocks.N.X` (the runtime's layout), quantizes every large 2D Linear
weight to int4 (group_size=64) and writes packed weights + bf16 group scales.

  out/int4-0000X-of-00002.safetensors : packed uint8 [out, in/2] + scales
  out/meta.json                       : group size, key list, dtype map
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("/home/ubuntu/jve-diff")
OUT = SRC / "int4"
OUT.mkdir(exist_ok=True)
GROUP = 64
MIN_NUMEL = 8_000_000  # quantize the big Linears only


def map_key(k: str) -> str:
    if k.startswith("double_blocks."):
        parts = k.split(".")
        if parts[2] == "attn":
            return ".".join(parts[:2] + parts[3:])
    return k


def quantize(w: torch.Tensor):
    """w: [out, in] -> (packed uint8 [out, in/2], scales bf16 [out, in/64])"""
    out_f, in_f = w.shape
    assert in_f % GROUP == 0 and in_f % 2 == 0, w.shape
    wr = w.float().reshape(out_f, in_f // GROUP, GROUP)
    scale = wr.abs().amax(dim=-1).clamp_min(1e-8) / 7.0
    q = torch.round(wr / scale.unsqueeze(-1)).clamp_(-8, 7).to(torch.int8)
    q = q.reshape(out_f, in_f) + 8                     # 0..15
    q8 = q.to(torch.uint8)
    packed = (q8[:, 0::2] | (q8[:, 1::2] << 4)).contiguous()
    return packed, scale.to(torch.bfloat16).contiguous()


def main() -> None:
    t0 = time.time()
    idx = json.load(open(SRC / "transformer_diffusion_pytorch_model.safetensors.index.json"))
    weight_map = idx["weight_map"]
    shards = sorted(set(weight_map.values()))
    print(f"{len(weight_map)} tensors in {len(shards)} shards", flush=True)

    # group keys by shard so we stream one shard at a time (RAM stays ~5 GB)
    per_shard: dict[str, list[str]] = {}
    for k, s in weight_map.items():
        per_shard.setdefault(s, []).append(k)

    packed_t: dict[str, torch.Tensor] = {}
    meta: dict[str, dict] = {}
    stats = []
    for si, shard in enumerate(shards, 1):
        path = SRC / "transformer" / shard
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in per_shard[shard]:
                t = f.get_tensor(k).to(torch.bfloat16)
                nk = map_key(k)
                if t.dim() == 2 and t.numel() >= MIN_NUMEL:
                    packed, scales = quantize(t)
                    packed_t[nk + ".qweight"] = packed
                    packed_t[nk + ".scales"] = scales
                    # reconstruction error
                    deq = (packed & 0x0F).to(torch.float32) - 8
                    deq2 = (packed >> 4).to(torch.float32) - 8
                    w_deq = torch.empty(t.shape, dtype=torch.float32)
                    w_deq[:, 0::2] = deq
                    w_deq[:, 1::2] = deq2
                    w_deq = (w_deq.reshape(t.shape[0], -1, GROUP) *
                             scales.float().unsqueeze(-1)).reshape(t.shape)
                    tf = t.float()
                    mse = (tf - w_deq).pow(2).mean().item()
                    var = tf.pow(2).mean().item()
                    cos = torch.nn.functional.cosine_similarity(
                        tf.flatten(), w_deq.flatten(), dim=0).item()
                    stats.append((nk, mse / max(var, 1e-12), cos, t.numel()))
                    meta[nk] = {"group": GROUP, "shape": list(t.shape),
                                "dtype": "int4", "rel_mse": mse / max(var, 1e-12)}
                else:
                    packed_t[nk] = t.contiguous()
                    meta[nk] = {"dtype": "bf16", "shape": list(t.shape)}
        print(f"  shard {si}/{len(shards)} done  tensors={len(packed_t)}  "
              f"{time.time() - t0:.0f}s", flush=True)

    # split the output into two files to keep each under ~5 GB
    keys = sorted(packed_t)
    half = len(keys) // 2
    parts = [keys[:half], keys[half:]]
    for pi, ks in enumerate(parts, 1):
        save_file({k: packed_t[k] for k in ks},
                  str(OUT / f"int4-{pi:05d}-of-{len(parts):05d}.safetensors"))
        print(f"wrote part {pi} ({len(ks)} tensors)", flush=True)

    json.dump({"group_size": GROUP, "tensors": meta},
              open(OUT / "meta.json", "w"), indent=1)

    n_q = len(stats)
    params_q = sum(s[3] for s in stats)
    rel = sorted(s[1] for s in stats)
    cosl = sorted(s[2] for s in stats)
    print(f"\nquantized {n_q} tensors, {params_q/1e9:.2f} B params")
    print(f"rel_mse  mean={sum(rel)/len(rel):.5f}  p50={rel[len(rel)//2]:.5f}  max={rel[-1]:.5f}")
    print(f"cosine   mean={sum(cosl)/len(cosl):.6f}  min={cosl[0]:.6f}")
    worst = sorted(stats, key=lambda s: -s[1])[:3]
    for nk, m, c, n in worst:
        print(f"   worst {nk:<48} rel_mse={m:.5f} cos={c:.5f}")
    tot = sum(p.numel() * p.element_size() for p in packed_t.values())
    print(f"output size {tot/2**30:.2f} GiB")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
