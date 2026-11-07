"""Low-VRAM mode helpers (fit the 480p24 streaming server on 32GB GPUs, e.g. RTX 5090).

Low-VRAM mode (JOYOMNI_LOW_VRAM=1):
  * builds the DiT on CPU and stages it to the GPU block-by-block, quantizing
    each block to FP8 as it lands, so the ~31GiB bf16 model is never GPU-resident;
  * keeps the text encoder in (pinned) CPU RAM via accelerate sequential offload,
    streaming through the GPU only during prompt encoding.
Measured at 480p24 under a 30GiB allocator cap: ~21.5GiB resident, ~28GiB peak
(see DEPLOYMENT.md).

Env knobs (all optional):
  JOYOMNI_LOW_VRAM=1|0          master switch (default 0; set 1 on ≤48GB cards)
  JOYOMNI_TE_OFFLOAD=1|0        text-encoder sequential CPU offload (default: follow low-VRAM)
  JOYOMNI_TE_PIN=1|0            pin the offloaded text-encoder weights for faster H2D (default: 1)
  JOYOMNI_VRAM_CAP_GB=<float>   testing hook: cap the caching allocator to emulate a smaller card
"""
from __future__ import annotations

import os

import torch

_TRUTHY = {"1", "true", "yes", "on"}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip().lower()


def vram_cap_bytes() -> int | None:
    raw = os.environ.get("JOYOMNI_VRAM_CAP_GB", "").strip()
    if not raw:
        return None
    return int(float(raw) * 2**30)


def low_vram_enabled() -> bool:
    return _env("JOYOMNI_LOW_VRAM") in _TRUTHY


def te_cpu_offload_enabled() -> bool:
    env = _env("JOYOMNI_TE_OFFLOAD")
    if not env:
        return low_vram_enabled()
    return env in _TRUTHY


def te_pin_memory_enabled() -> bool:
    return _env("JOYOMNI_TE_PIN", "1") in _TRUTHY


def apply_vram_cap_for_testing(device: torch.device | str | int | None = None) -> None:
    """Clamp the CUDA caching allocator to JOYOMNI_VRAM_CAP_GB (testing hook).

    Lets a big card (e.g. 96GB Pro 6000) emulate a 32GB RTX 5090: allocations
    beyond the cap raise OOM exactly like they would on the smaller card.
    """
    cap = vram_cap_bytes()
    if cap is None or not torch.cuda.is_available():
        return
    device_obj = torch.device(device if device is not None else "cuda")
    if device_obj.type != "cuda":
        return
    index = device_obj.index if device_obj.index is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(index).total_memory
    fraction = min(1.0, cap / total)
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    print(
        f"#####[LOW-VRAM] allocator capped at {cap / 2**30:.1f} GiB "
        f"({fraction:.3f} of {total / 2**30:.1f} GiB) on cuda:{index} for testing",
        flush=True,
    )


def log_mode() -> None:
    print(
        f"#####[LOW-VRAM] mode={'ON' if low_vram_enabled() else 'off'} "
        f"te_cpu_offload={te_cpu_offload_enabled()}",
        flush=True,
    )
