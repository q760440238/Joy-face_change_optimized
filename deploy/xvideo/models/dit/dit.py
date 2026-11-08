from contextlib import contextmanager
import logging
import os
import time
from typing import Callable, Dict, Iterable, List, Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from xvideo.fp8_status import format_fp8_precision_status, fp8_precision_status


from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps


try:
    from flash_attn.cute import flash_attn_func as _fa4_func
    _FA4_IMPORTED = True
except Exception:  # noqa: BLE001
    # flash-attn-4 is optional. On GPUs/toolchains where it isn't available
    # (e.g. no FA4 build for this arch), fall back to SDPA (cuDNN/flash) at
    # runtime instead of failing at import time.
    _fa4_func = None
    _FA4_IMPORTED = False


SOURCE_ID_TARGET = 0.0
SOURCE_ID_EDIT_CONDITION = 1.0
SOURCE_ID_EXTRA_REF_IMAGE = 2.0

TIME_FREQ_DIM = 256

NORM_EPS = 1e-6
NUM_MODULATION_CHUNKS = 6

SELF_ATTN_MODE_REF_IMAGE_CACHE = "ref_image_cache"


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


_FP8_IMG_ENABLED = _env_on("JOYOMNI_FP8_IMG")
_FP8_TXT_ENABLED = _env_on("JOYOMNI_FP8_TXT")


def _fp8_stream_wanted(stream: str) -> bool:
    if stream == "img":
        return _FP8_IMG_ENABLED
    if stream == "txt":
        return _FP8_TXT_ENABLED
    return False


def _maybe_install_fp8_stream(block, stream: str) -> None:
    """Quantize the attn qkv/proj + mlp up/down Linears of one stream ("img" or "txt")
    to FP8 and free the bf16 weights (~31GiB across the model). Idempotent per
    (block, stream); the FP8 twins fully replace the bf16 Linears in forward."""
    if not _fp8_stream_wanted(stream):
        return
    installed_flag = f"_fp8_{stream}_installed"
    if getattr(block, installed_flag, False):
        return
    from xvideo.models.dit.fp8_linear import Fp8Linear
    setattr(block, f"_fp8_{stream}_attn_qkv",
            Fp8Linear.from_linear(getattr(block, f"{stream}_attn_qkv")))
    setattr(block, f"_fp8_{stream}_attn_proj",
            Fp8Linear.from_linear(getattr(block, f"{stream}_attn_proj")))
    ff = getattr(block, f"{stream}_mlp")
    up_lin = down_lin = None
    gelu_mod = None
    for m in ff.net:
        if isinstance(m, nn.Linear) and down_lin is None and up_lin is not None:
            down_lin = m
            continue
        if isinstance(m, nn.Linear) and up_lin is None:
            up_lin = m
            continue
        if hasattr(m, "proj") and isinstance(m.proj, nn.Linear) and up_lin is None:
            up_lin = m.proj
            gelu_mod = m
    if up_lin is None or down_lin is None:
        raise RuntimeError(f"could not locate mlp up/down Linears in {ff}")
    setattr(block, f"_fp8_{stream}_mlp_up", Fp8Linear.from_linear(up_lin))
    setattr(block, f"_fp8_{stream}_mlp_down", Fp8Linear.from_linear(down_lin))
    _approx = getattr(gelu_mod, "approximate", "tanh") if gelu_mod is not None else "tanh"
    setattr(block, f"_{stream}_mlp_act",
            lambda x, _a=_approx: torch.nn.functional.gelu(x, approximate=_a))
    # Biases stay (the Fp8Linear twins share the same tensors); only the
    # weights are freed. state_dict() of these Linears is dead from here on.
    for lin in (getattr(block, f"{stream}_attn_qkv"),
                getattr(block, f"{stream}_attn_proj"),
                up_lin, down_lin):
        lin.weight.data = lin.weight.data.new_empty(0)
    setattr(block, installed_flag, True)


def _fp8_stream_enabled(block, stream: str) -> bool:
    return bool(getattr(block, f"_fp8_{stream}_installed", False))


def fp8_precision_report(transformer, *, base_precision: str) -> tuple[dict, str]:
    """Return DiT Linear FP8 state and its runtime summary."""
    status = fp8_precision_status(
        transformer,
        {stream: _fp8_stream_wanted(stream) for stream in ("img", "txt")},
        base_precision=base_precision,
    )
    return status, format_fp8_precision_status(status)



from xvideo.models.dit import sgl_fused_ops as _sgl_fused

# --- text-stream KV capture (for stripping the txt stream at inference) -----
# Populated only while a capture is active: layer_idx -> (txt_k, txt_v).
_TEXT_KV_SINK: dict | None = None


def begin_text_kv_capture() -> dict:
    global _TEXT_KV_SINK
    _TEXT_KV_SINK = {}
    return _TEXT_KV_SINK


def end_text_kv_capture() -> dict | None:
    global _TEXT_KV_SINK
    sink, _TEXT_KV_SINK = _TEXT_KV_SINK, None
    return sink

# Installed by the pipeline for the duration of one denoise step:
# layer_idx -> (txt_k, txt_v), already qk-normed. None disables the fast path.
_TEXT_KV_ACTIVE: dict | None = None


def set_text_kv_active(cache: dict | None) -> None:
    global _TEXT_KV_ACTIVE
    _TEXT_KV_ACTIVE = cache


# --- text-stream offload (keep the txt weights in host RAM) -----------------
# Only a text-on forward touches these: the per-session K/V capture and a full
# (non-streaming) sample. The streaming server runs every denoise/store forward
# with skip_text_stream=True, so once the K/V cache exists the weights can live
# on the host and be paged back in one block at a time for the next capture.
_IMG_STREAM_ATTRS = (
    "img_mod",
    "img_norm1",
    "img_norm2",
    "img_attn_q_norm",
    "img_attn_k_norm",
    "img_attn_qkv",
    "img_attn_proj",
    "img_mlp",
    "_fp8_img_attn_qkv",
    "_fp8_img_attn_proj",
    "_fp8_img_mlp_up",
    "_fp8_img_mlp_down",
)

_TEXT_STREAM_ATTRS = (
    "txt_mod",
    "txt_norm1",
    "txt_norm2",
    "txt_attn_q_norm",
    "txt_attn_k_norm",
    "txt_attn_qkv",
    "txt_attn_proj",
    "txt_mlp",
    "_fp8_txt_attn_qkv",
    "_fp8_txt_attn_proj",
    "_fp8_txt_mlp_up",
    "_fp8_txt_mlp_down",
)


def _text_stream_modules(block):
    for name in _TEXT_STREAM_ATTRS:
        module = getattr(block, name, None)
        if isinstance(module, nn.Module):
            yield module


def _module_tensors(module):
    return list(module.parameters()) + list(module.buffers())


def text_stream_bytes(transformer) -> int:
    """Bytes held by every block's text-stream weights."""
    return sum(
        tensor.numel() * tensor.element_size()
        for block in transformer.double_blocks
        for module in _text_stream_modules(block)
        for tensor in _module_tensors(module)
    )


def move_block_text_stream(block, device) -> int:
    """Move one block's text-stream weights to `device`; returns bytes moved."""
    moved = 0
    for module in _text_stream_modules(block):
        for tensor in _module_tensors(module):
            moved += tensor.numel() * tensor.element_size()
        module.to(device)
    return moved


def _stream_tensor_chains(block, attrs):
    """Yield (attr_chain, is_param, tensor) for every leaf tensor of `attrs`."""
    for root in attrs:
        module = getattr(block, root, None)
        if not isinstance(module, nn.Module):
            continue
        for path, sub in module.named_modules():
            steps = tuple(path.split(".")) if path else ()
            for is_param, table in ((True, sub._parameters), (False, sub._buffers)):
                for name, tensor in list(table.items()):
                    if tensor is None:
                        continue
                    yield (root, *steps, name), is_param, tensor


def _text_tensor_chains(block):
    return _stream_tensor_chains(block, _TEXT_STREAM_ATTRS)


def _resolve_chain(block, chain):
    obj = block
    for step in chain[:-1]:
        obj = getattr(obj, step)
    return obj, chain[-1]


def _bind_tensor(obj, name, tensor, is_param: bool) -> None:
    if is_param:
        obj._parameters[name] = nn.Parameter(tensor, requires_grad=False)
    else:
        obj._buffers[name] = tensor


class _TextStreamPager:
    """Keep the text weights in host RAM and page them in one block at a time.

    One GPU slot per tensor (shared by all blocks, they are identical), one
    pinned host mirror per block. Page-in/page-out are plain `copy_` calls on the
    current stream, so they are ordered with the block's kernels.
    """

    def __init__(self, transformer, device: torch.device):
        blocks = list(transformer.double_blocks)
        template = list(_text_tensor_chains(blocks[0]))
        self.device = device
        self.blocks = blocks
        self.specs = [(chain, is_param) for chain, is_param, _ in template]
        self.slots = [
            torch.empty_strided(tensor.size(), tensor.stride(), dtype=tensor.dtype, device=device)
            for _, _, tensor in template
        ]
        self.hosts: list[list[torch.Tensor]] = []
        pinned = 0
        for block in blocks:
            hosts = []
            for chain, is_param in self.specs:
                obj, name = _resolve_chain(block, chain)
                tensor = (obj._parameters if is_param else obj._buffers)[name]
                if not tensor.is_pinned():
                    try:
                        tensor = tensor.pin_memory()
                        _bind_tensor(obj, name, tensor, is_param)
                        pinned += 1
                    except Exception:  # noqa: BLE001 - pinned memory is best effort
                        pass
                hosts.append(tensor)
            self.hosts.append(hosts)
        self.pinned_tensors = pinned
        self.bytes = sum(t.numel() * t.element_size() for t in self.slots)

    def _bind(self, layer_idx: int, tensors) -> None:
        block = self.blocks[layer_idx]
        for (chain, is_param), tensor in zip(self.specs, tensors):
            obj, name = _resolve_chain(block, chain)
            _bind_tensor(obj, name, tensor, is_param)

    def page_in(self, layer_idx: int) -> None:
        hosts = self.hosts[layer_idx]
        for slot, host in zip(self.slots, hosts):
            slot.copy_(host, non_blocking=True)
        self._bind(layer_idx, self.slots)

    def page_out(self, layer_idx: int) -> None:
        # Inference never writes the weights, so the host mirrors are still
        # authoritative: rebinding is enough, no copy-back needed.
        self._bind(layer_idx, self.hosts[layer_idx])


_STREAM_PREFETCH_ENV = "JOYOMNI_STREAM_PREFETCH"


def stream_prefetch_enabled() -> bool:
    """Double-buffer the streamed blocks (experimental, see the commit message)."""
    return os.environ.get(_STREAM_PREFETCH_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


class _BlockPager:
    """Page one image block in at a time; the host copy is authoritative.

    Inference never writes the weights, so no page-out copy is needed - the next
    page-in simply overwrites the slots.
    """

    def __init__(self, transformer, device: torch.device):
        blocks = list(transformer.double_blocks)
        template = list(_stream_tensor_chains(blocks[0], _IMG_STREAM_ATTRS))
        self.device = device
        self.blocks = blocks
        self.specs = [(chain, is_param) for chain, is_param, _ in template]
        self.slots = [
            torch.empty_strided(tensor.size(), tensor.stride(), dtype=tensor.dtype, device=device)
            for _, _, tensor in template
        ]
        self.hosts: list[list[torch.Tensor]] = []
        pinned = 0
        for block in blocks:
            hosts = []
            for chain, is_param in self.specs:
                obj, name = _resolve_chain(block, chain)
                tensor = (obj._parameters if is_param else obj._buffers)[name]
                if not tensor.is_pinned():
                    try:
                        tensor = tensor.pin_memory()
                        pinned += 1
                    except Exception:  # noqa: BLE001 - best effort
                        pass
                hosts.append(tensor)
            self.hosts.append(hosts)
        # All blocks share the slots; every forward pages the block it is about to
        # run over them, so the modules can stay bound to the slots permanently.
        for block in blocks:
            for (chain, is_param), slot in zip(self.specs, self.slots):
                obj, name = _resolve_chain(block, chain)
                _bind_tensor(obj, name, slot, is_param)
        self._bound_once = True
        self.pinned_tensors = pinned
        self.bytes = sum(t.numel() * t.element_size() for t in self.slots)
        self.prefetch_enabled = stream_prefetch_enabled()
        if self.prefetch_enabled:
            self._buffers = [self.slots, self._alloc_buffer()]
            self._copy_stream = torch.cuda.Stream(device=device)
            self._ready = [torch.cuda.Event(), torch.cuda.Event()]
            self._free: list[torch.cuda.Event | None] = [None, None]
            self._buffer = 0
            self.bytes *= 2
        else:
            self._buffers = [self.slots]
            self._buffer = 0

    def _alloc_buffer(self) -> list[torch.Tensor]:
        return [
            torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, device=self.device)
            for t in self.slots
        ]

    def prefetch(self, layer_idx: int) -> None:
        """Start copying `layer_idx` into the idle buffer on the copy stream."""
        buf = 1 - self._buffer
        wait = self._free[buf]
        if wait is not None:
            self._copy_stream.wait_event(wait)
        with torch.cuda.stream(self._copy_stream):
            for slot, host in zip(self._buffers[buf], self.hosts[layer_idx]):
                slot.copy_(host, non_blocking=True)
            self._ready[buf].record(self._copy_stream)

    def use(self, layer_idx: int) -> None:
        """Wait for the prefetched buffer and bind the block to it."""
        buf = 1 - self._buffer
        torch.cuda.current_stream().wait_event(self._ready[buf])
        block = self.blocks[layer_idx]
        for (chain, is_param), slot in zip(self.specs, self._buffers[buf]):
            obj, name = _resolve_chain(block, chain)
            _bind_tensor(obj, name, slot, is_param)
        self._buffer = buf

    def release(self) -> None:
        """Mark the buffer currently in use free once the enqueued work drains."""
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self._free[self._buffer] = event

    def page_in(self, layer_idx: int) -> None:
        for slot, host in zip(self.slots, self.hosts[layer_idx]):
            slot.copy_(host, non_blocking=True)


def _ensure_block_pager(transformer, device) -> "_BlockPager":
    pager = getattr(transformer, "_block_stream_pager", None)
    if pager is None:
        started = time.time()
        pager = _BlockPager(transformer, device)
        transformer._block_stream_pager = pager
        print(
            f"#####[BLOCK-PAGER] {pager.bytes / 2**20:.0f} MiB of GPU slots for "
            f"{len(pager.blocks)} blocks, built in {time.time() - started:.2f}s",
            flush=True,
        )
    return pager


def _ensure_text_pager(transformer, device) -> "_TextStreamPager":
    pager = getattr(transformer, "_text_stream_pager", None)
    if pager is None:
        started = time.time()
        pager = _TextStreamPager(transformer, device)
        pager.blocks = transformer.double_blocks
        transformer._text_stream_pager = pager
        print(
            f"#####[TEXT-PAGER] {pager.bytes / 2**20:.0f} MiB of GPU slots, "
            f"{len(pager.hosts)} blocks, {pager.pinned_tensors} tensors pinned, "
            f"built in {time.time() - started:.2f}s",
            flush=True,
        )
    return pager

def move_text_stream(transformer, device, pin: bool = True) -> int:
    """Move the whole text stream to `device` (host RAM by default).

    Off-loading also pins the host copy so the per-block page-in during the next
    K/V capture runs at PCIe speed instead of pageable-RAM speed.
    """
    device = torch.device(device)
    # Any wholesale move invalidates the persistent slot/host mirrors.
    transformer._text_stream_pager = None
    moved = 0
    for block in transformer.double_blocks:
        moved += move_block_text_stream(block, device)
    transformer._text_stream_offloaded = device.type != "cuda"
    if device.type == "cpu" and pin:
        for block in transformer.double_blocks:
            for module in _text_stream_modules(block):
                for tensor in _module_tensors(module):
                    if tensor.is_pinned() or not tensor.is_floating_point():
                        continue
                    try:
                        pinned = tensor.data.pin_memory()
                    except Exception:  # noqa: BLE001 - pinnable memory is best effort
                        break
                    if pinned.stride() != tensor.stride():
                        continue
                    tensor.data = pinned
    return moved


from xvideo.models.dit.rope import apply_rotary_emb, get_1d_rotary_pos_embed


class ModulateWan(nn.Module):
    def __init__(self, hidden_size: int, factor: int, dtype=None, device=None):
        super().__init__()
        self.factor = factor
        self.modulate_table = nn.Parameter(
            torch.randn(1, factor, hidden_size,
                        dtype=dtype, device=device) / hidden_size**0.5,
            requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 3:
            x = x.unsqueeze(1)
        return [o.squeeze(1) for o in (self.modulate_table + x).chunk(self.factor, dim=1)]


logger = logging.getLogger(__name__)


def _clone_kv_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.detach().clone()


_FA4_DISABLED = not _FA4_IMPORTED or os.environ.get("JOYOMNI_FA4_ATTN", "1") == "0"

_SDPA_BACKENDS = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]

_SAGE_ATTN_ENABLED = _env_on("JOYOMNI_SAGE_ATTN")
_sage_attn_func = None
if _SAGE_ATTN_ENABLED:
    try:
        from sageattention import sageattn as _sage_attn_func
    except Exception as _sage_exc:  # noqa: BLE001
        _sage_attn_func = None
        print(f"[dit] JOYOMNI_SAGE_ATTN=1 but sageattention unavailable: {_sage_exc!r}")


def _sdpa_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if _sage_attn_func is not None:
        return _sage_attn_func(query, key, value, tensor_layout="NHD", is_causal=False)
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    with sdpa_kernel(_SDPA_BACKENDS, set_priority=True):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return out.transpose(1, 2)


def _flash_attention4(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    global _FA4_DISABLED
    original_dtype = query.dtype
    if query.dtype not in (torch.float16, torch.bfloat16):
        query = query.to(torch.bfloat16)
        key = key.to(torch.bfloat16)
        value = value.to(torch.bfloat16)
    if not _FA4_DISABLED:
        try:
            out = _fa4_func(query, key, value, causal=False)
            if isinstance(out, tuple):
                out = out[0]
            return out.to(original_dtype)
        except Exception as exc:  # noqa: BLE001
            _FA4_DISABLED = True
            logger.warning(f"[{__name__}] FA4 unavailable, falling back to SDPA (flash/cuDNN): {exc!r}")
    return _sdpa_attention(query, key, value).to(original_dtype)


def attention_backend() -> str:
    if not _FA4_DISABLED:
        return "FA4"
    if _sage_attn_func is not None:
        return "sage"
    return "SDPA (cuDNN/flash)"


_FP8_HEAD_ENV = "JOYOMNI_FP8_HEAD"


def install_fp8_head(transformer) -> int:
    """Quantise the head/embedding Linears to FP8 (JOYOMNI_FP8_HEAD=1).

    Covers `proj_out` and all of `condition_embedder` (time_proj, time_embedder,
    text_embedder: 152 M params). `img_in` keeps its 5-D weight and stays bf16.
    """
    if os.environ.get(_FP8_HEAD_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}:
        return 0
    if getattr(transformer, "_fp8_head_installed", False):
        return 0
    from xvideo.models.dit.fp8_linear import Fp8Linear

    embedder = transformer.condition_embedder
    targets = (
        (transformer, "proj_out"),
        (embedder, "time_proj"),
        (embedder.time_embedder, "linear_1"),
        (embedder.time_embedder, "linear_2"),
        (embedder.text_embedder, "linear_1"),
        (embedder.text_embedder, "linear_2"),
    )
    moved = 0
    for module, name in targets:
        linear = getattr(module, name, None)
        if not isinstance(linear, nn.Linear):
            continue
        moved += sum(p.numel() * p.element_size() for p in linear.parameters())
        setattr(module, name, Fp8Linear.from_linear(linear))
    transformer._fp8_head_installed = True
    return moved


_KV_DTYPE_ENV = "JOYOMNI_KV_DTYPE"
_KV_FP8_ENV = "JOYOMNI_KV_FP8"
_KV_FP8_MAX = 448.0
_KV_INT8_MAX = 127.0


def kv_cache_dtype() -> str:
    """Storage dtype of the cached image K/V: "bf16" (default), "fp8" or "int8"."""
    raw = os.environ.get(_KV_DTYPE_ENV, "").strip().lower()
    if not raw and os.environ.get(_KV_FP8_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        raw = "fp8"
    return raw if raw in {"fp8", "int8"} else "bf16"


def kv_fp8_enabled() -> bool:
    """Backwards-compatible alias for the older boolean FP8 knob."""
    return kv_cache_dtype() == "fp8"


def kv_entry_from_tensors(
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
    *,
    pre_rope: bool,
) -> Dict[str, object]:
    """Build one cache entry, quantised to FP8 when JOYOMNI_KV_FP8 is on."""
    entry: Dict[str, object] = {"pre_rope": bool(pre_rope)}
    if key is None or value is None:
        entry["key"] = None
        entry["value"] = None
        return entry
    dtype = kv_cache_dtype()
    if dtype == "bf16":
        entry["key"] = _clone_kv_tensor(key)
        entry["value"] = _clone_kv_tensor(value)
        return entry
    quant_max = _KV_FP8_MAX if dtype == "fp8" else _KV_INT8_MAX
    for name, tensor in (("key", key), ("value", value)):
        tensor = tensor.detach().to(torch.float32)
        scale = (tensor.abs().amax(dim=-1, keepdim=True) / quant_max).clamp_min(1e-8)
        scaled = (tensor / scale).clamp(-quant_max, quant_max)
        if dtype == "fp8":
            entry[f"{name}_q"] = scaled.to(torch.float8_e4m3fn)
        else:
            entry[f"{name}_q"] = scaled.round().to(torch.int8)
        entry[f"{name}_scale"] = scale
    entry["quant"] = dtype
    return entry


def kv_entry_tensors(
    entry: Dict[str, object],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Materialise an entry as (key, value) on `device` (dequantising if needed)."""
    key = entry.get("key")
    value = entry.get("value")
    if key is not None and value is not None:
        return key.to(device=device, dtype=dtype), value.to(device=device, dtype=dtype)
    key_q = entry.get("key_q")
    value_q = entry.get("value_q")
    if key_q is None or value_q is None:
        return None, None
    key = (
        key_q.to(device=device, dtype=torch.float32)
        * entry["key_scale"].to(device=device, dtype=torch.float32)
    ).to(dtype)
    value = (
        value_q.to(device=device, dtype=torch.float32)
        * entry["value_scale"].to(device=device, dtype=torch.float32)
    ).to(dtype)
    return key, value


def _concat_kv_entries(
    entries: Iterable[Dict[str, torch.Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    cached_freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    keys = []
    values = []
    pre_rope_offset = 0

    for entry in entries:
        if entry is None:
            continue
        key, value = kv_entry_tensors(entry, device=device, dtype=dtype)
        if key is None or value is None:
            continue

        if entry.get("pre_rope", False) and cached_freqs_cis is not None:
            cos_all, sin_all = cached_freqs_cis
            seg_len = key.shape[1]
            cos_seg = cos_all[..., pre_rope_offset: pre_rope_offset + seg_len, :]
            sin_seg = sin_all[..., pre_rope_offset: pre_rope_offset + seg_len, :]
            key = apply_rotary_emb(key, (cos_seg, sin_seg))
            pre_rope_offset += seg_len

        keys.append(key)
        values.append(value)

    if not keys:
        return None, None

    return torch.cat(keys, dim=1), torch.cat(values, dim=1)


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = NORM_EPS,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


class MMDoubleStreamBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu-approximate",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateWan(hidden_size, NUM_MODULATION_CHUNKS, **factory_kwargs)
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=NORM_EPS, **factory_kwargs
        )

        self.img_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=True, **factory_kwargs
        )
        self.img_attn_q_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=NORM_EPS, **factory_kwargs)
        self.img_attn_k_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=NORM_EPS, **factory_kwargs)
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=True, **factory_kwargs
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=NORM_EPS, **factory_kwargs
        )
        self.img_mlp = FeedForward(hidden_size, inner_dim=mlp_hidden_dim,
                                   activation_fn=mlp_act_type)

        self.txt_mod = ModulateWan(hidden_size, NUM_MODULATION_CHUNKS, **factory_kwargs)
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=NORM_EPS, **factory_kwargs
        )

        self.txt_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=True, **factory_kwargs
        )
        self.txt_attn_q_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=NORM_EPS, **factory_kwargs)
        self.txt_attn_k_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=NORM_EPS, **factory_kwargs)
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=True, **factory_kwargs
        )

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=NORM_EPS, **factory_kwargs
        )
        self.txt_mlp = FeedForward(hidden_size, inner_dim=mlp_hidden_dim,
                                   activation_fn=mlp_act_type)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        vis_freqs_cis: tuple = None,
        kv_cache_reader: Optional[Callable[[Optional[int]], Iterable[Dict[str, torch.Tensor]]]] = None,
        kv_cache_writer: Optional[Callable[[Optional[int], torch.Tensor, torch.Tensor], None]] = None,
        kv_cache_assembler: Optional[Callable[..., Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]] = None,
        layer_idx: Optional[int] = None,
        skip_text_stream: bool = False,
        kv_cache_pre_rope: bool = False,
        cached_freqs_cis: Optional[tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _maybe_install_fp8_stream(self, "img")
        _fp8_on = _fp8_stream_enabled(self, "img")
        if not skip_text_stream:
            _maybe_install_fp8_stream(self, "txt")
        _fp8_txt_on = (not skip_text_stream) and _fp8_stream_enabled(self, "txt")
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec)
        if not skip_text_stream:
            (
                txt_mod1_shift,
                txt_mod1_scale,
                txt_mod1_gate,
                txt_mod2_shift,
                txt_mod2_scale,
                txt_mod2_gate,
            ) = self.txt_mod(vec)

        img_modulated = _sgl_fused.fused_layernorm_modulate(
            img, shift=img_mod1_shift, scale=img_mod1_scale,
            weight=self.img_norm1.weight if self.img_norm1.elementwise_affine else None,
            bias=self.img_norm1.bias if self.img_norm1.elementwise_affine else None,
            eps=self.img_norm1.eps,
        )
        img_qkv = self._fp8_img_attn_qkv(img_modulated) if _fp8_on else self.img_attn_qkv(img_modulated)
        _iv = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.heads_num, -1)
        img_q, img_k, img_v = _iv[:, :, 0], _iv[:, :, 1], _iv[:, :, 2]
        if kv_cache_pre_rope:
            img_k_for_cache = _sgl_fused.rmsnorm_qk_bf16(
                img_k, self.img_attn_k_norm.weight, eps=self.img_attn_k_norm.eps
            )
        img_q, img_k = _sgl_fused.fused_qk_norm_rope_3d(
            img_q, img_k,
            q_norm_weight=self.img_attn_q_norm.weight,
            k_norm_weight=self.img_attn_k_norm.weight,
            freqs_cis=vis_freqs_cis,
            eps=self.img_attn_q_norm.eps,
        )
        if not kv_cache_pre_rope:
            img_k_for_cache = img_k

        if not skip_text_stream:
            txt_modulated = _sgl_fused.fused_layernorm_modulate(
                txt, shift=txt_mod1_shift, scale=txt_mod1_scale,
                weight=self.txt_norm1.weight if self.txt_norm1.elementwise_affine else None,
                bias=self.txt_norm1.bias if self.txt_norm1.elementwise_affine else None,
                eps=self.txt_norm1.eps,
            )
            txt_qkv = self._fp8_txt_attn_qkv(txt_modulated) if _fp8_txt_on else self.txt_attn_qkv(txt_modulated)
            _tv = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.heads_num, -1)
            txt_q, txt_k, txt_v = _tv[:, :, 0], _tv[:, :, 1], _tv[:, :, 2]
            txt_q = self.txt_attn_q_norm(txt_q).to(txt_v.dtype)
            txt_k = self.txt_attn_k_norm(txt_k).to(txt_v.dtype)
            if _TEXT_KV_SINK is not None:
                _TEXT_KV_SINK[layer_idx] = (txt_k.detach(), txt_v.detach())

        if skip_text_stream:
            q = img_q
            k = img_k
            v = img_v
        else:
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)
            v = torch.cat((img_v, txt_v), dim=1)

        if kv_cache_writer is not None:
            kv_cache_writer(layer_idx, img_k_for_cache, img_v)

        if kv_cache_assembler is not None:
            cached_key, cached_value = kv_cache_assembler(
                layer_idx,
                device=q.device,
                dtype=q.dtype,
                cached_freqs_cis=cached_freqs_cis if kv_cache_pre_rope else None,
            )
        elif kv_cache_reader is not None:
            cached_key, cached_value = _concat_kv_entries(
                kv_cache_reader(layer_idx),
                device=q.device,
                dtype=q.dtype,
                cached_freqs_cis=cached_freqs_cis if kv_cache_pre_rope else None,
            )
        else:
            cached_key = cached_value = None

        if cached_key is not None:
            k = torch.cat([cached_key, k], dim=1)
            v = torch.cat([cached_value, v], dim=1)
        if skip_text_stream and _TEXT_KV_ACTIVE is not None:
            _tkv = _TEXT_KV_ACTIVE.get(layer_idx)
            if _tkv is not None:
                k = torch.cat([k, _tkv[0].to(dtype=k.dtype, device=k.device)], dim=1)
                v = torch.cat([v, _tkv[1].to(dtype=v.dtype, device=v.device)], dim=1)

        attn = _flash_attention4(q, k, v)
        attn = attn.flatten(2, 3)
        if skip_text_stream:
            img_attn = attn
        else:
            img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1]:]

        def _img_proj_call(x):
            return self._fp8_img_attn_proj(x) if _fp8_on else self.img_attn_proj(x)

        def _img_mlp_call(x):
            if _fp8_on:
                h = self._fp8_img_mlp_up(x)
                h = self._img_mlp_act(h)
                return self._fp8_img_mlp_down(h)
            return self.img_mlp(x)

        def _txt_proj_call(x):
            return self._fp8_txt_attn_proj(x) if _fp8_txt_on else self.txt_attn_proj(x)

        def _txt_mlp_call(x):
            if _fp8_txt_on:
                h = self._fp8_txt_mlp_up(x)
                h = self._txt_mlp_act(h)
                return self._fp8_txt_mlp_down(h)
            return self.txt_mlp(x)

        def _txt_tail(txt_in):
            t = _sgl_fused.fused_add_gate(
                txt_in, _txt_proj_call(txt_attn),
                txt_mod1_gate,
            )
            t_mod2 = _sgl_fused.fused_layernorm_modulate(
                t, shift=txt_mod2_shift, scale=txt_mod2_scale,
                weight=self.txt_norm2.weight if self.txt_norm2.elementwise_affine else None,
                bias=self.txt_norm2.bias if self.txt_norm2.elementwise_affine else None,
                eps=self.txt_norm2.eps,
            )
            return _sgl_fused.fused_add_gate(
                t, _txt_mlp_call(t_mod2),
                txt_mod2_gate,
            )

        img = _sgl_fused.fused_add_gate(
            img, _img_proj_call(img_attn),
            img_mod1_gate,
        )
        img_mod2_modulated = _sgl_fused.fused_layernorm_modulate(
            img, shift=img_mod2_shift, scale=img_mod2_scale,
            weight=self.img_norm2.weight if self.img_norm2.elementwise_affine else None,
            bias=self.img_norm2.bias if self.img_norm2.elementwise_affine else None,
            eps=self.img_norm2.eps,
        )
        img = _sgl_fused.fused_add_gate(
            img, _img_mlp_call(img_mod2_modulated),
            img_mod2_gate,
        )

        if not skip_text_stream:
            txt = _txt_tail(txt)

        return img, txt


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(
        self,   
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        timestep_shape = timestep.shape
        timestep_is_sequence = timestep.ndim > 1
        if timestep_is_sequence:
            timestep = timestep.flatten()
        timestep = self.timesteps_proj(timestep)

        # FP8 Linears keep their weights in buffers, so `parameters()` can be
        # empty and a float8 dtype is not a valid timestep dtype.
        _time_embedder_dtype = next(iter(self.time_embedder.parameters()), None)
        time_embedder_dtype = (
            _time_embedder_dtype.dtype if _time_embedder_dtype is not None else torch.bfloat16
        )
        if time_embedder_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            time_embedder_dtype = torch.bfloat16
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))
        if timestep_is_sequence:
            timestep_proj = timestep_proj.reshape(*timestep_shape, timestep_proj.shape[-1])

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        return timestep_proj, encoder_hidden_states


class Transformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):

    @register_to_config
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        text_states_dim: int = 4096,
        mlp_width_ratio: float = 4.0,
        mm_double_blocks_depth: int = 20,
        rope_dim_list: List[int] = [16, 56, 56],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        theta: int = 256,
        chunk_size: Optional[int] = None,
        local_window_size: int = 4,
        global_sink_chunk: bool = True,
        causal: bool = False,
        use_inference_kv_cache: bool = False,
        source_id_rope_dim: int = 128,
        source_id_rope_theta: float = 256.0,
    ):
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"`chunk_size` must be positive when provided, got {chunk_size}.")
        if local_window_size <= 0:
            raise ValueError(f"`local_window_size` must be positive, got {local_window_size}.")
        if causal and chunk_size is None:
            raise ValueError("`chunk_size` must be provided when `causal=True`.")
        self.out_channels = out_channels or in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.rope_dim_list = rope_dim_list
        self.theta = theta
        self.chunk_size = chunk_size
        self.local_window_size = local_window_size
        self.global_sink_chunk = global_sink_chunk
        self.causal = causal
        self._initial_use_inference_kv_cache = use_inference_kv_cache

        self.source_id_rope_dim = int(source_id_rope_dim)
        self.source_id_rope_theta = float(source_id_rope_theta)

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = hidden_size
        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )

        self.img_in = nn.Conv3d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=hidden_size,
            time_freq_dim=TIME_FREQ_DIM,
            time_proj_dim=hidden_size * NUM_MODULATION_CHUNKS,
            text_embed_dim=text_states_dim,
        )

        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        self.norm_out = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=NORM_EPS
        )
        self.proj_out = nn.Linear(
            hidden_size, self.out_channels * math.prod(patch_size),
            **factory_kwargs)

        self._ensure_kv_cache_state()
        self._use_inference_kv_cache = use_inference_kv_cache

    def _ensure_kv_cache_state(self) -> None:
        if not hasattr(self, "_inference_kv_cache"):
            self._inference_kv_cache = {"cond": {}, "uncond": {}}
        if not hasattr(self, "_use_inference_kv_cache"):
            self._use_inference_kv_cache = bool(getattr(self, "_initial_use_inference_kv_cache", False))
        if not hasattr(self, "_kv_cache_mode"):
            self._kv_cache_mode = None
        if not hasattr(self, "_kv_cache_scope"):
            self._kv_cache_scope = None
        if not hasattr(self, "_kv_cache_chunk_id"):
            self._kv_cache_chunk_id = None
        if not hasattr(self, "_kv_cache_selected_chunk_ids"):
            self._kv_cache_selected_chunk_ids = None
        if not hasattr(self, "_kv_cache_pre_rope"):
            self._kv_cache_pre_rope = False
        if not hasattr(self, "_kv_cache_generation"):
            self._kv_cache_generation = 0
        if not hasattr(self, "_kv_assembly_version"):
            self._kv_assembly_version = None
        if not hasattr(self, "_kv_assembly_cache"):
            self._kv_assembly_cache = {}
        if not hasattr(self, "_kv_assembly_freqs"):
            self._kv_assembly_freqs = None
        if not hasattr(self, "_text_stream_offloaded"):
            self._text_stream_offloaded = False
        if not hasattr(self, "_text_stream_paged"):
            self._text_stream_paged = False

    def reset_inference_kv_cache(self) -> None:
        self._ensure_kv_cache_state()
        self._inference_kv_cache = {"cond": {}, "uncond": {}}
        self._use_inference_kv_cache = bool(getattr(self.config, "use_inference_kv_cache", False))
        self._kv_cache_mode = None
        self._kv_cache_scope = None
        self._kv_cache_chunk_id = None
        self._kv_cache_selected_chunk_ids = None
        self._kv_cache_pre_rope = False
        self._kv_cache_generation = 0
        self._kv_assembly_version = None
        self._kv_assembly_cache = {}
        self._kv_assembly_freqs = None

    def configure_inference_kv_cache(
        self,
        *,
        scope: Optional[str],
        mode: Optional[str],
        chunk_id: Optional[int] = None,
        selected_chunk_ids: Optional[list[int]] = None,
        pre_rope: bool = False,
    ) -> None:
        self._ensure_kv_cache_state()
        self._kv_cache_scope = scope
        self._kv_cache_mode = mode
        self._kv_cache_chunk_id = chunk_id
        self._kv_cache_selected_chunk_ids = list(selected_chunk_ids) if selected_chunk_ids is not None else None
        self._use_inference_kv_cache = mode in {"reuse", "store", "reuse_store"}
        self._kv_cache_pre_rope = bool(pre_rope)

    @contextmanager
    def cache_context(self, scope: Optional[str]):
        self._ensure_kv_cache_state()
        previous_scope = self._kv_cache_scope
        self._kv_cache_scope = scope
        try:
            yield self
        finally:
            self._kv_cache_scope = previous_scope

    def _get_cache_scope_store(self) -> Optional[Dict[int, Dict[int, Dict[str, torch.Tensor]]]]:
        self._ensure_kv_cache_state()
        if self._kv_cache_scope is None:
            return None
        return self._inference_kv_cache.setdefault(self._kv_cache_scope, {})

    def _read_layer_kv_cache(self, layer_idx: Optional[int]) -> list[Dict[str, torch.Tensor]]:
        scope_store = self._get_cache_scope_store()
        if scope_store is None or layer_idx is None:
            return []
        selected_chunk_ids = self._kv_cache_selected_chunk_ids or []
        layer_entries = []
        for selected_chunk_id in selected_chunk_ids:
            chunk_store = scope_store.get(selected_chunk_id)
            if chunk_store is None:
                continue
            entry = chunk_store.get(layer_idx)
            if entry is not None:
                layer_entries.append(entry)
        return layer_entries

    def _assemble_layer_kv_cache_cached(
        self,
        layer_idx: Optional[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        cached_freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if layer_idx in self._kv_assembly_cache:
            return self._kv_assembly_cache[layer_idx]
        entries = self._read_layer_kv_cache(layer_idx)
        cached_key, cached_value = _concat_kv_entries(
            entries, device=device, dtype=dtype, cached_freqs_cis=cached_freqs_cis
        )
        self._kv_assembly_cache[layer_idx] = (cached_key, cached_value)
        return cached_key, cached_value

    def _write_layer_kv_cache(
        self,
        layer_idx: Optional[int],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        scope_store = self._get_cache_scope_store()
        if scope_store is None or layer_idx is None or self._kv_cache_chunk_id is None:
            return
        chunk_store = scope_store.setdefault(self._kv_cache_chunk_id, {})
        chunk_store[layer_idx] = kv_entry_from_tensors(
            key, value, pre_rope=bool(self._kv_cache_pre_rope)
        )
        self._kv_cache_generation += 1

    def evict_kv_cache_chunks(self, chunk_ids_to_keep: set[int]) -> None:
        self._ensure_kv_cache_state()
        for scope in ("cond", "uncond"):
            scope_store = self._inference_kv_cache.get(scope)
            if scope_store is None:
                continue
            evict_ids = [cid for cid in scope_store if cid not in chunk_ids_to_keep]
            for cid in evict_ids:
                del scope_store[cid]
            if evict_ids:
                self._kv_cache_generation += 1

    def get_rotary_pos_embed_from_ids(
        self,
        *,
        frame_ids: torch.Tensor,
        spatial_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        post_patch_height, post_patch_width = spatial_shape
        device = frame_ids.device
        temporal_positions = frame_ids.to(dtype=torch.float32)
        spatial_tokens_per_frame = post_patch_height * post_patch_width
        if temporal_positions.numel() % spatial_tokens_per_frame != 0:
            raise ValueError(
                f"`frame_ids` length {temporal_positions.numel()} is not divisible by spatial token count {spatial_tokens_per_frame}."
            )

        h_positions = torch.arange(post_patch_height, dtype=torch.float32, device=device)
        w_positions = torch.arange(post_patch_width, dtype=torch.float32, device=device)
        h_grid, w_grid = torch.meshgrid(h_positions, w_positions, indexing="ij")
        h_positions = h_grid.reshape(-1).repeat(temporal_positions.numel() // spatial_tokens_per_frame)
        w_positions = w_grid.reshape(-1).repeat(temporal_positions.numel() // spatial_tokens_per_frame)

        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if sum(rope_dim_list) != head_dim:
            raise ValueError("sum(rope_dim_list) should equal to head_dim of attention layer")

        cos_list = []
        sin_list = []
        for dim, positions in zip(rope_dim_list, (temporal_positions, h_positions, w_positions)):
            cos, sin = get_1d_rotary_pos_embed(
                dim,
                positions,
                theta=self.theta,
            )
            cos_list.append(cos)
            sin_list.append(sin)
        vis_freqs = (torch.cat(cos_list, dim=1), torch.cat(sin_list, dim=1))

        return vis_freqs

    def generate_source_id_rope(
        self,
        source_id: torch.Tensor,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        role_dim = max(0, min(int(self.source_id_rope_dim), int(head_dim)))

        half_head = head_dim // 2
        cos_half = torch.ones(*source_id.shape, half_head, device=device, dtype=torch.float32)
        sin_half = torch.zeros(*source_id.shape, half_head, device=device, dtype=torch.float32)

        inv_freq = 1.0 / (
            self.source_id_rope_theta **
            (torch.arange(0, role_dim, 2, device=device, dtype=torch.float32) / role_dim)
        )

        angles = source_id.unsqueeze(-1) * inv_freq
        cos_half[..., : role_dim // 2] = torch.cos(angles)
        sin_half[..., : role_dim // 2] = torch.sin(angles)
        return (
            cos_half.repeat_interleave(2, dim=-1).to(dtype=dtype),
            sin_half.repeat_interleave(2, dim=-1).to(dtype=dtype),
        )

    @staticmethod
    def _get_patch_shape(latent: torch.Tensor, patch_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        _, _, num_frames, height, width = latent.shape
        return (
            num_frames // patch_size[0],
            height // patch_size[1],
            width // patch_size[2],
        )

    @staticmethod
    def _get_token_frame_ids(
        post_patch_shape: Tuple[int, int, int],
        device: torch.device,
        temporal_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_frames, post_patch_height, post_patch_width = post_patch_shape
        spatial_tokens_per_frame = post_patch_height * post_patch_width
        if temporal_ids is None:
            frame_ids = torch.arange(num_frames, device=device, dtype=torch.long)
        else:
            frame_ids = torch.as_tensor(temporal_ids, device=device, dtype=torch.long)
            if frame_ids.ndim != 1 or frame_ids.numel() != num_frames:
                raise ValueError(
                    f"`temporal_ids` must be 1D with length {num_frames}, got {tuple(frame_ids.shape)}."
                )
        return frame_ids.repeat_interleave(spatial_tokens_per_frame)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        ref_video_latent: Optional[torch.Tensor] = None,
        current_temporal_ids: Optional[torch.Tensor] = None,
        cached_temporal_ids: Optional[torch.Tensor] = None,
        kv_cache_mode: Optional[str] = None,
        kv_cache_scope: Optional[str] = None,
        kv_cache_chunk_id: Optional[int] = None,
        kv_cache_selected_chunk_ids: Optional[list[int]] = None,
        kv_cache_pre_rope: bool = False,
        self_attn_input_mode: Optional[str] = None,
        skip_text_stream: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        self._ensure_kv_cache_state()
        self.configure_inference_kv_cache(
            scope=kv_cache_scope,
            mode=kv_cache_mode,
            chunk_id=kv_cache_chunk_id,
            selected_chunk_ids=kv_cache_selected_chunk_ids,
            pre_rope=kv_cache_pre_rope,
        )

        batch_size = hidden_states.shape[0]
        patch_size = tuple(self.patch_size)
        current_patch_shape = self._get_patch_shape(hidden_states, patch_size)
        current_seq_len = math.prod(current_patch_shape)
        device = hidden_states.device

        hidden_tokens = self.img_in(hidden_states).flatten(2).transpose(1, 2)
        temporal_ids = None
        if current_temporal_ids is not None:
            current_temporal_ids = torch.as_tensor(current_temporal_ids, device=device, dtype=torch.long)
            assert current_temporal_ids.shape == (batch_size, current_patch_shape[0]), (
                f"`current_temporal_ids` must have shape {(batch_size, current_patch_shape[0])}, "
                f"got {tuple(current_temporal_ids.shape)}."
            )
            temporal_ids = current_temporal_ids[0]
        if self_attn_input_mode == SELF_ATTN_MODE_REF_IMAGE_CACHE:
            current_source_id = torch.full((current_seq_len,), SOURCE_ID_EXTRA_REF_IMAGE, device=device, dtype=torch.float32)
        else:
            current_source_id = torch.full((current_seq_len,), SOURCE_ID_TARGET, device=device, dtype=torch.float32)
        current_frame_ids = self._get_token_frame_ids(current_patch_shape, device, temporal_ids=temporal_ids)
        current_rotary = self.get_rotary_pos_embed_from_ids(
            frame_ids=current_frame_ids,
            spatial_shape=(current_patch_shape[1], current_patch_shape[2]),
        )

        latent_segments = [hidden_tokens]
        rotary_segments = [current_rotary]
        source_id_segments = [current_source_id]

        if ref_video_latent is not None:
            if ref_video_latent.shape[0] != batch_size:
                raise ValueError(
                    f"Ref video latent batch size {ref_video_latent.shape[0]} does not match hidden states batch size {batch_size}."
                )
            ref_video_patch_shape = self._get_patch_shape(ref_video_latent, patch_size)
            if ref_video_patch_shape[1:] != current_patch_shape[1:]:
                raise ValueError(
                    "Ref video latent spatial patch shape must match noisy latent spatial patch shape: "
                    f"{ref_video_patch_shape[1:]} != {current_patch_shape[1:]}."
                )
            ref_video_tokens = self.img_in(ref_video_latent).flatten(2).transpose(1, 2)
            video_frame_ids = self._get_token_frame_ids(ref_video_patch_shape, device, temporal_ids=temporal_ids)
            latent_segments.append(ref_video_tokens)
            rotary_segments.append(self.get_rotary_pos_embed_from_ids(
                frame_ids=video_frame_ids,
                spatial_shape=(ref_video_patch_shape[1], ref_video_patch_shape[2]),
            ))
            source_id_segments.append(
                torch.full((ref_video_tokens.shape[1],), SOURCE_ID_EDIT_CONDITION, device=device, dtype=torch.float32)
            )

        img = torch.cat(latent_segments, dim=1)
        visual_source_id = torch.cat(source_id_segments, dim=0).unsqueeze(0)
        current_indices = torch.arange(current_seq_len, device=device).unsqueeze(0)
        vis_freqs_cis = (
            torch.cat([rotary[0] for rotary in rotary_segments], dim=0).unsqueeze(0),
            torch.cat([rotary[1] for rotary in rotary_segments], dim=0).unsqueeze(0),
        )

        head_dim = self.hidden_size // self.heads_num
        cos_3d, sin_3d = vis_freqs_cis
        cos_role, sin_role = self.generate_source_id_rope(
            source_id=visual_source_id,
            head_dim=head_dim,
            device=cos_3d.device,
            dtype=cos_3d.dtype,
        )
        new_cos = cos_3d * cos_role - sin_3d * sin_role
        new_sin = sin_3d * cos_role + cos_3d * sin_role
        vis_freqs_cis = (new_cos, new_sin)

        vec, txt = self.condition_embedder(timestep, encoder_hidden_states)
        vec = vec.unflatten(-1, (NUM_MODULATION_CHUNKS, -1))

        use_memo = bool(kv_cache_pre_rope and
                        cached_temporal_ids is not None and
                        kv_cache_mode in {"reuse", "reuse_store"})
        _assembly_stamp = None
        if use_memo:
            _cached_ids_src = torch.as_tensor(cached_temporal_ids, dtype=torch.long)
            _temporal_sig = (tuple(_cached_ids_src.shape), tuple(_cached_ids_src.reshape(-1).tolist()))
            _assembly_stamp = (
                self._kv_cache_scope,
                tuple(self._kv_cache_selected_chunk_ids or ()),
                bool(self._kv_cache_pre_rope),
                _temporal_sig,
                self._kv_cache_generation,
            )

        cached_freqs_cis = None
        if use_memo and _assembly_stamp == self._kv_assembly_version:
            cached_freqs_cis = self._kv_assembly_freqs
        elif kv_cache_pre_rope and cached_temporal_ids is not None:
            cached_ids_tensor = torch.as_tensor(cached_temporal_ids, device=device, dtype=torch.long)
            cached_frame_ids = self._get_token_frame_ids(
                (cached_ids_tensor.shape[1], current_patch_shape[1], current_patch_shape[2]),
                device,
                temporal_ids=cached_ids_tensor[0],
            )
            cached_vis_freqs = self.get_rotary_pos_embed_from_ids(
                frame_ids=cached_frame_ids,
                spatial_shape=(current_patch_shape[1], current_patch_shape[2]),
            )
            cached_freqs_cis = (
                cached_vis_freqs[0].unsqueeze(0),
                cached_vis_freqs[1].unsqueeze(0),
            )
            if use_memo:
                self._kv_assembly_version = _assembly_stamp
                self._kv_assembly_freqs = cached_freqs_cis
                self._kv_assembly_cache = {}

        _graph_assembler = getattr(self, "_graph_kv_assembler", None)
        _graph_writer = getattr(self, "_graph_kv_writer", None)
        _text_paged = (
            self._text_stream_offloaded
            and self._text_stream_paged
            and not skip_text_stream
        )
        if self._text_stream_offloaded and not skip_text_stream and not _text_paged:
            raise RuntimeError(
                "text stream is offloaded to host: run with skip_text_stream=True and a "
                "text K/V cache, or set transformer._text_stream_paged=True to page it in"
            )
        _text_pager = _ensure_text_pager(self, device) if _text_paged else None
        _block_pager = (
            _ensure_block_pager(self, device) if getattr(self, "_stream_blocks", False) else None
        )
        for layer_idx, block in enumerate(self.double_blocks):
            if _block_pager is not None:
                if _block_pager.prefetch_enabled:
                    if layer_idx == 0:
                        _block_pager.prefetch(0)
                    else:
                        _block_pager.release()
                        _block_pager.prefetch(layer_idx)
                    _block_pager.use(layer_idx)
                else:
                    _block_pager.page_in(layer_idx)
            if _text_pager is not None:
                _text_pager.page_in(layer_idx)
            img, txt = block(
                img,
                txt,
                vec,
                vis_freqs_cis,
                layer_idx=layer_idx,
                kv_cache_reader=None if _graph_assembler is not None else (
                    self._read_layer_kv_cache if kv_cache_mode in {"reuse", "reuse_store"} else None),
                kv_cache_writer=_graph_writer if _graph_writer is not None else (
                    self._write_layer_kv_cache if kv_cache_mode in {"store", "reuse_store"} else None),
                kv_cache_assembler=_graph_assembler if _graph_assembler is not None else (
                    self._assemble_layer_kv_cache_cached if use_memo else None),
                skip_text_stream=skip_text_stream,
                kv_cache_pre_rope=kv_cache_pre_rope,
                cached_freqs_cis=cached_freqs_cis,
            )
            if _text_pager is not None:
                _text_pager.page_out(layer_idx)
        if _block_pager is not None and _block_pager.prefetch_enabled:
            _block_pager.release()

        img = self.proj_out(self.norm_out(img))

        gather_index = current_indices.unsqueeze(-1).expand(-1, -1, img.shape[-1])
        img = torch.gather(img, dim=1, index=gather_index)
        img = self.unpatchify(img, current_patch_shape[0], current_patch_shape[1], current_patch_shape[2])

        return (img, txt)

    def unpatchify(self, x, t, h, w):
        c = self.out_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs
