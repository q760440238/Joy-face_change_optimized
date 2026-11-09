import os

import torch
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration
from loguru import logger

from xvideo.models.dit import Transformer3DModel
from xvideo.models.vae import XVAEChunkCausal
from xvideo.models.scheduler import FlowMatchDiscreteScheduler
from xvideo.models.pipeline import Pipeline, PRECISION_TO_TYPE


def _commit_gb() -> str:
    """Process private commit + RSS (psutil), for memory-bound debugging."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        return (
            f"commit={info.private / 2**30:.1f}GiB rss={info.rss / 2**30:.1f}GiB "
            f"sysfree={psutil.virtual_memory().available / 2**30:.1f}GiB"
        )
    except Exception:  # noqa: BLE001
        return "(psutil missing)"


def load_text_encoder(
    text_encoder_ckpt: str,
    device: torch.device = torch.device("cpu"),
    torch_dtype: torch.dtype = torch.bfloat16,
    cpu_offload: bool = False,
):
    print(f"#####[TE] before from_pretrained {_commit_gb()}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        text_encoder_ckpt,
        dtype=torch_dtype,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    print(f"#####[TE] after from_pretrained {_commit_gb()}", flush=True)
    if cpu_offload:
        # The 7B text encoder runs only at session start / KV reset, so it stays
        # in CPU RAM and streams through the GPU during encode_prompt (~0 resident).
        model = model.eval().requires_grad_(False)
        from accelerate import cpu_offload as _accelerate_cpu_offload
        from xvideo.lowvram import te_pin_memory_enabled

        state_dict = None
        if te_pin_memory_enabled():
            try:
                state_dict = {
                    name: (tensor.pin_memory() if tensor.device.type == "cpu" else tensor)
                    for name, tensor in model.state_dict().items()
                }
                logger.info("Text encoder offload: pinned host copy for faster H2D staging")
            except RuntimeError as exc:
                logger.warning(f"Text encoder offload: pin_memory failed ({exc!r}); using pageable RAM")
                state_dict = None
        print(f"#####[TE] before cpu_offload {_commit_gb()}", flush=True)
        _accelerate_cpu_offload(
            model,
            execution_device=torch.device(device),
            state_dict=state_dict,
        )
        print(f"#####[TE] after cpu_offload {_commit_gb()}", flush=True)
        logger.info(f"Text encoder: sequential CPU offload enabled (execution device {device})")
    else:
        model = model.to(device).eval().requires_grad_(False)
    tokenizer = Qwen2Tokenizer.from_pretrained(
        text_encoder_ckpt,
        local_files_only=True,
    )
    return tokenizer, model


def _arch_params(arch_config) -> dict:
    return dict(arch_config.get("params", {})) if isinstance(arch_config, dict) else {}


def build_vae(cfg, device: torch.device):
    pretrained = cfg.vae_arch_config["pretrained"]
    vae = XVAEChunkCausal.from_pretrained(
        pretrained,
        torch_dtype=PRECISION_TO_TYPE[cfg.vae_precision],
        device=device,
        **_arch_params(cfg.vae_arch_config),
    )
    return vae


def load_pipeline(cfg, dit, device: torch.device):
    from xvideo.lowvram import te_cpu_offload_enabled

    vae = build_vae(cfg, device)

    te_offload = te_cpu_offload_enabled()
    tokenizer, text_encoder = load_text_encoder(
        torch_dtype=PRECISION_TO_TYPE[cfg.text_encoder_precision],
        device=device,
        cpu_offload=te_offload,
        **_arch_params(cfg.text_encoder_arch_config),
    )

    scheduler = FlowMatchDiscreteScheduler(**_arch_params(cfg.scheduler_arch_config))

    pipeline = Pipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=dit,
        scheduler=scheduler,
        args=cfg,
        **_arch_params(cfg.pipeline_arch_config),
    )
    if te_offload:
        # DiffusionPipeline.to() refuses to move pipelines that contain a
        # sequentially-offloaded model; place the remaining modules manually
        # (the DiT is already on `device` from load_dit).
        pipeline.vae.to(device)
        return pipeline
    return pipeline.to(device)


def _move_dit_to_device_lowvram(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Stage a CPU-built DiT to `device` block-by-block, quantizing to FP8 (and
    releasing bf16) as each block lands, so the ~31GiB bf16 model is never
    resident on the GPU (peak ~= FP8 footprint + one bf16 block)."""
    import time

    from xvideo.models.dit.dit import (
        _fp8_stream_wanted,
        _maybe_install_fp8_stream,
        move_block_text_stream,
    )
    from xvideo.lowvram import text_offload_enabled

    fp8_streams = [s for s in ("img", "txt") if _fp8_stream_wanted(s)]
    if len(fp8_streams) < 2:
        logger.warning(
            "Low-VRAM DiT load without full FP8 quantization "
            f"(enabled streams: {fp8_streams or 'none'}): non-quantized blocks stay "
            "bf16 and a 32GB card will likely OOM. Set JOYOMNI_FP8_IMG=1 and "
            "JOYOMNI_FP8_TXT=1."
        )

    for name, child in model.named_children():
        if name == "double_blocks":
            continue
        child.to(device)

    started = time.time()
    num_blocks = len(model.double_blocks)
    offload_text = text_offload_enabled() and "txt" in fp8_streams
    text_bytes = 0
    for i, block in enumerate(model.double_blocks):
        block.to(device)
        for stream in fp8_streams:
            _maybe_install_fp8_stream(block, stream)
        if offload_text:
            # Park the text-stream weights in host RAM as each block lands: the
            # text branch is only needed for the per-session K/V capture, which
            # pages one block back in at a time.
            text_bytes += move_block_text_stream(block, "cpu")
        if (i + 1) % 10 == 0 or i + 1 == num_blocks:
            logger.info(
                f"Low-VRAM DiT load: {i + 1}/{num_blocks} blocks staged to {device} "
                f"({time.time() - started:.1f}s)"
            )
    if offload_text:
        model._text_stream_offloaded = True
        model._text_stream_paged = True
        logger.info(
            f"Low-VRAM DiT load: {text_bytes / 2**30:.2f} GiB text-stream weights kept in host RAM"
        )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        free_b, total_b = torch.cuda.mem_get_info(device)
        logger.info(
            f"Low-VRAM DiT load done in {time.time() - started:.1f}s; "
            f"device used {(total_b - free_b) / 2**30:.1f}/{total_b / 2**30:.1f} GiB"
        )
    return model



def _load_int8_inplace(model, directory, collect_blocks: bool = False, store=None):
    """Dequantise an int8 export straight into `model` (no full bf16 state dict).

    Shards are split by tensor name, so a `.q8weight` and its `.q8scales` can live
    in different files: index every tensor first, then pair them up.
    """
    import json
    from pathlib import Path

    from safetensors import safe_open

    root = Path(directory)
    state = model.state_dict()
    where = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                where[key] = path
        print(f"#####[INT8-LOAD] indexed {path.name}", flush=True)

    quantised = sorted(k[: -len(".q8weight")] for k in where if k.endswith(".q8weight"))
    copied = 0
    store = {} if store is None else store
    for base in quantised:
        if base not in state:
            continue
        if collect_blocks and base.startswith("double_blocks."):
            layer = int(base.split(".")[1])
            # descriptor only: the shard stays mmapped and is read when paged in
            store.setdefault(layer, {})[base.split(".", 2)[2]] = (
                str(where[base + ".q8weight"]),
                base + ".q8weight",
                str(where[base + ".q8scales"]),
                base + ".q8scales",
            )
            if len(store) % 4 == 0 and sum(len(v) for v in store.values()) % 40 == 0:
                print(f"#####[INT8-LOAD] collect {sum(len(v) for v in store.values())} tensors", flush=True)
            continue
        with safe_open(str(where[base + ".q8weight"]), framework="pt") as fq:
            q = fq.get_tensor(base + ".q8weight")
        with safe_open(str(where[base + ".q8scales"]), framework="pt") as fs:
            sc = fs.get_tensor(base + ".q8scales")
        state[base].copy_((q.float() * sc.float()).to(torch.bfloat16))
        copied += 1
        if copied % 100 == 0:
            print(f"#####[INT8-LOAD] {copied}/894 tensors", flush=True)

    for key, path in where.items():
        if key.endswith(".q8weight") or key.endswith(".q8scales") or key not in state:
            continue
        if collect_blocks and key.startswith("double_blocks."):
            layer = int(key.split(".")[1])
            with safe_open(str(path), framework="pt") as f:
                store.setdefault(layer, {})[key.split(".", 2)[2]] = f.get_tensor(key).clone()
            continue
        with safe_open(str(path), framework="pt") as f:
            state[key].copy_(f.get_tensor(key))
        copied += 1

    del state
    logger.info(f"int8 stream load: {copied} tensors dequantised straight into the model")
    if collect_blocks:
        logger.info(f"int8 lazy blocks: {len(store)} layers kept int8 in host RAM")
    return store


def load_dit(cfg, device: torch.device) -> torch.nn.Module:
    from xvideo.lowvram import low_vram_enabled

    device_obj = torch.device(device)
    low_vram = low_vram_enabled() and device_obj.type == "cuda"

    state_dict = None
    _int4_dir = os.environ.get("JOYOMNI_DIT_INT4_DIR", "").strip()
    _stream_load = os.environ.get("JOYOMNI_INT8_STREAM_LOAD", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if _int4_dir and not _stream_load:
        from xvideo.models.dit.int4_linear import load_int4_state_dict
        logger.info(f"Loading int4 DiT from {_int4_dir} (dequantised to bf16)")
        state_dict = load_int4_state_dict(
            _int4_dir,
            mlp_bf16_dir=os.environ.get("JOYOMNI_INT4_KEEP_MLP_BF16") or None,
        )
        logger.info(f"int4 state_dict ready: {len(state_dict)} tensors")
    elif os.environ.get("JOYOMNI_DIT_SD_DIR", "").strip():
        from xvideo.models.dit.int4_linear import load_safetensors_state_dict
        _sd_dir = os.environ["JOYOMNI_DIT_SD_DIR"].strip()
        logger.info(f"Loading DiT from safetensors dir {_sd_dir}")
        state_dict = load_safetensors_state_dict(_sd_dir)
        logger.info(f"safetensors state_dict ready: {len(state_dict)} tensors")
    elif cfg.dit_ckpt is not None and not (_int4_dir and _stream_load):
        logger.info(f"Loading DiT checkpoint: {cfg.dit_ckpt}")
        # No mmap on HF Spaces: /data is FUSE, page faults crawl (hours for 30 GiB).
        mmap_ok = "SPACE_ID" not in os.environ
        state_dict = torch.load(cfg.dit_ckpt, map_location="cpu", weights_only=True, mmap=mmap_ok)
        if "model" in state_dict:
            state_dict = state_dict["model"]

    dtype = PRECISION_TO_TYPE[cfg.dit_precision]
    # Low-VRAM: build + fill on CPU, then stage blocks to the GPU as FP8 (the
    # bf16 model alone would already overflow a 32GB card).
    build_device = torch.device("cpu") if low_vram else device
    _lazy_blocks = os.environ.get("JOYOMNI_INT8_LAZY_BLOCKS", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if _lazy_blocks:
        # blocks are paged from the int8 store: never allocate their bf16 copies
        build_device = torch.device("meta")
    if build_device.type == "meta":
        with torch.device("meta"):
            model = Transformer3DModel(
                dtype=dtype, device=build_device, **_arch_params(cfg.dit_arch_config)
            )
    else:
        model = Transformer3DModel(
            dtype=dtype, device=build_device, **_arch_params(cfg.dit_arch_config)
        )
    model.to(device=build_device)
    if build_device.type == "meta":
        # materialise only the tensors that are not paged (embeddings, head, ...)
        for mod_name, module in model.named_modules():
            if mod_name == "double_blocks" or mod_name.startswith("double_blocks."):
                continue
            for n, prm in list(module._parameters.items()):
                if prm is not None:
                    module._parameters[n] = torch.nn.Parameter(
                        torch.empty(prm.shape, dtype=prm.dtype, device="cpu"), requires_grad=False
                    )
            for n, buf in list(module._buffers.items()):
                if buf is not None:
                    module._buffers[n] = torch.empty(buf.shape, dtype=buf.dtype, device="cpu")
        print("#####[DIT] meta skeleton materialised (non-block tensors only)", flush=True)
        # the constructor leaves a few params in float32; the forward runs bf16
        with torch.no_grad():
            for name, prm in model.named_parameters():
                if not name.startswith("double_blocks.") and prm.dtype != dtype:
                    prm.data = prm.data.to(dtype)
            for name, buf in model.named_buffers():
                if not name.startswith("double_blocks.") and buf.is_floating_point() and buf.dtype != dtype:
                    buf.data = buf.data.to(dtype)

    if state_dict is not None or (_int4_dir and _stream_load):
        load_state_dict = None
        for prefix in ("model.", "module.", "transformer."):
            if state_dict is not None and any(k.startswith(prefix) for k in state_dict):
                state_dict = {
                    (k[len(prefix):] if k.startswith(prefix) else k): v
                    for k, v in state_dict.items()
                }

        load_state_dict = {} if state_dict is not None else None
        for k, v in (state_dict or {}).items():
            if (
                k == "img_in.weight" and
                hasattr(model, "img_in") and
                model.img_in.weight.shape != v.shape
            ):
                logger.info(f"Inflate {k} from {v.shape} to {model.img_in.weight.shape}")
                v = v.reshape_as(v.new_zeros(model.img_in.weight.shape))
            load_state_dict[k] = v

        print(f"#####[DIT] before load {_commit_gb()}", flush=True)
        missing_keys, unexpected_keys = [], []
        _lazy_blocks = os.environ.get("JOYOMNI_INT8_LAZY_BLOCKS", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if _int4_dir and _stream_load:
            if _lazy_blocks:
                # Bind the block tensors to GPU slots FIRST: that drops the bf16
                # host copies (32.5 GiB) before the int8 store (15 GiB) is read,
                # so the two never coexist.
                from xvideo.models.dit.dit import _ensure_block_pager, _ensure_text_pager

                _store: dict = {}
                model._int8_block_store = _store
                _ensure_text_pager(model, device_obj)
                _ensure_block_pager(model, device_obj)
                print("#####[INT8-LAZY] pagers bound, collecting int8 store", flush=True)
                _load_int8_inplace(model, _int4_dir, collect_blocks=True, store=_store)
                print("#####[INT8-LAZY] int8 store ready", flush=True)
                logger.info("int8 lazy blocks: block tensors rebound to GPU slots")
            else:
                _load_int8_inplace(model, _int4_dir)
        if load_state_dict is not None:
            missing_keys, unexpected_keys = model.load_state_dict(load_state_dict, strict=True)
        if missing_keys:
            logger.warning(f"Missing keys when loading DiT: {missing_keys[:20]}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading DiT: {unexpected_keys[:20]}")

    print(f"#####[DIT] after load {_commit_gb()}", flush=True)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Instantiate model with {total_params / 1e9:.2f}B parameters")

    param_dtypes = {param.dtype for param in model.parameters()}
    if len(param_dtypes) > 1:
        # NB: model.to(dtype) needs a second full copy (~32 GB for the DiT), which
        # does not fit next to the loaded weights on a 64 GB box. Cast in place.
        logger.warning(f"Model has mixed dtypes: {param_dtypes}. Casting in place to {dtype}")
        with torch.no_grad():
            for _, param in model.named_parameters():
                if param.dtype != dtype:
                    param.data = param.data.to(dtype)
            for _, buf in model.named_buffers():
                if buf.is_floating_point() and buf.dtype != dtype:
                    buf.data = buf.data.to(dtype)

    stream_dit = low_vram and os.environ.get("JOYOMNI_STREAM_DIT", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if stream_dit:
        # Keep every block in host RAM: quantise in place on CPU and let the
        # forward page one block in at a time (~0.19 GiB resident instead of 7.8).
        from xvideo.models.dit.dit import _fp8_stream_wanted, _maybe_install_fp8_stream

        for name, child in model.named_children():
            if name == "double_blocks":
                continue
            child.to(device_obj)
        for block in [] if getattr(model, "_int8_block_store", None) else model.double_blocks:
            # quantise on the GPU then park: the CUDA FP8 cast rounds slightly
            # differently from the CPU one, and the mismatch compounds over 40
            # layers (measured 28.6 dB against the resident build).
            block.to(device_obj)
            for stream_name in ("img", "txt"):
                if _fp8_stream_wanted(stream_name):
                    _maybe_install_fp8_stream(block, stream_name)
            block.to("cpu")
        model._stream_blocks = True
        model._text_stream_offloaded = True
        model._text_stream_paged = True
        torch.cuda.empty_cache()
        logger.info(
            "Streaming DiT: blocks stay in host RAM (JOYOMNI_STREAM_DIT=1); "
            "one block is paged in per forward"
        )
    elif low_vram:
        model = _move_dit_to_device_lowvram(model, device_obj)

    from xvideo.models.dit.dit import install_fp8_head
    head_bytes = install_fp8_head(model)
    if head_bytes:
        logger.info(
            f"FP8 head: {head_bytes / 2**20:.0f} MiB of head/embedding weights quantised"
        )

    return model.eval()


__all__ = [
    "load_dit",
    "load_pipeline",
    "build_vae",
]
