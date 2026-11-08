# Joy-face_change_optimized

[JoyAI-Video-Edit](https://github.com/jd-opensource/JoyAI-Video-Edit) 的优化分支（Apache-2.0）。
目标：**降低部署门槛**——可移植算子、量化工具链、以及实测出来的精度/显存结论。

- GitHub: https://github.com/q760440238/Joy-face_change_optimized
- Hugging Face: https://huggingface.co/kylinfires/Joy-face_change_optimized
- 魔搭: https://www.modelscope.cn/models/kylinfires/Joy-face_change_optimized

## 本分支做了什么

| 文件 | 作用 |
|---|---|
| `deploy/xvideo/models/dit/fused_ops_torch.py` | 用纯 PyTorch 复刻 `joyomni_ops` 的 CUDA kernel（LayerNorm+modulate、QK-norm+3D RoPE、RMSNorm） |
| `deploy/xvideo/models/dit/sgl_fused_ops.py` | `joyomni_ops` 不可用时自动回退到上面的实现（ROCm / CPU 也能跑） |
| `deploy/xvideo/models/dit/int4_linear.py` | int4（分组）与 int8（逐通道）的 weight-only Linear，含加载器 |
| `deploy/xvideo/models/models.py` | 新增 `JOYOMNI_DIT_SD_DIR`（bf16 分片）与 `JOYOMNI_DIT_INT4_DIR`（量化权重）两条加载路径 |
| `tools/quantization/` | RTN int4 (gs=64)、int8、attn-int8/MLP-int4 (gs=16) 三套量化脚本 |
| `tools/jve_client.py` | 无头 WebSocket 驱动（参考图 + 指令 + 帧序列），用于自动化验证 |

PyTorch 回退与 CUDA 版**逐张量验证一致**：99.997% 元素完全相同，最大偏差 1 个 bf16 ulp。

## 量化实测结论（840×480，同权重同素材同 prompt）

| 精度 | 权重体积 | 与 bf16 的 PSNR | 参考人物相似度 |
|---|---:|---:|---:|
| bf16（基线） | 31.8 GB | — | 0.314 |
| **int8（逐通道）** | 15.2 GiB | 21.8 dB | **0.359** ✅ |
| int4 (gs=64) | 8.1 GiB | 19.0 dB | 0.103 ❌ |
| attn int8 + MLP int4 (gs=16) | 11.4 GiB | 19.2 dB | 0.199 ⚠️ |

流水线跨重启完全确定（同一份 int4 权重在两个新进程里跑出 **138 dB**，身份相似度 0.103 vs 0.103），所以上述差异是真实损失而非测量噪声。

**结论：int4 会摧毁"参考图换人"能力，int8 不会。** 身份信息完全走注意力通路——只把注意力压到 int4、MLP 全保 bf16，换人也会失效（0.027）。

## 优化方向

### 1. 剥离文本流权重（最大收益，进行中）

模型每层的参数构成：

| 分支 | 每 block | ×40 | 占比 |
|---|---:|---:|---:|
| 图像流 `img_attn + img_mlp` | 201 M | 8.05 B | 49.5% |
| **文本流 `txt_attn + txt_mlp`** | **201 M** | **8.05 B** | **49.5%** |

文本 token 的 K/V 只依赖指令，**与画面无关**，一次会话内指令不变。因此可以：

1. 会话开始时跑**一次**文本分支，逐层缓存 `txt_k`（qk-norm 之后）、`txt_v`
2. 之后每个 chunk 传 `skip_text_stream=True`，attention 的 K/V = 当前图像 + 静态缓存（文本 + 参考图）
3. 预填充完成后把 **8.05 B 文本流权重卸到 CPU 内存**

代码基础已经具备：`dit.py` 的 block forward 本来就有 `skip_text_stream` 参数，`pipeline._prefill_static_reference_kv_cache()` 已经用同一套 KV 缓存机制给参考图做静态预填充。

| 组成 | 现状 | 剥离后 |
|---|---:|---:|
| 图像流权重 | 8.05 GB | 8.05 GB |
| 文本流权重 | 8.05 GB | **0**（常驻内存） |
| VAE + graph + KV | 7.7 GB | 7.7 GB |
| **合计** | **23.9 GB** | **~15.8 GB** |

即 **16 GB 显卡可跑**，且用 int8（质量 0.359 优于 bf16 的 0.314），不靠 int4 硬压。额外收益：每 chunk 少算 40 次 `txt_mlp`，帧率预计 +30~40%。

> 状态：方案与代码路径已确认，实现待完成。验收标准：缓存版 vs 完整版 PSNR > 40 dB、身份相似度差 < 0.01。

### 2. 其他方向

- **MLP 5-bit**：注意力 int8 + MLP 5-bit 约 13.3 GiB，误差预计 0.35%，相似度有望到 0.27–0.29
- **参考图 KV 复用**：同一张参考图跨会话复用，省一次 prefill
- **VAE 分时驻留**：encode/decode 之间释放对方，省约 0.7 GB
- **HEAD 输出量化**：`proj_out` 与 `img_in` 保持 bf16（已默认）

## 快速开始

```bash
python -m pip install -r deploy/requirements.txt

# 三种权重加载方式（互不影响）
# 1) 原始 .pth（默认）
# 2) bf16 分片
export JOYOMNI_DIT_SD_DIR=/path/to/diffusers/transformer
# 3) 量化权重
export JOYOMNI_DIT_INT4_DIR=/path/to/int8_export

bash deploy/run_server.sh
```

量化：

```bash
python tools/quantization/quantize_int8.py     # 15.2 GiB，质量可用
python tools/quantization/quantize_int4.py     # 8.1 GiB，会摧毁换人能力
```

---

## English (concise)

Fork of [JoyAI-Video-Edit](https://github.com/jd-opensource/JoyAI-Video-Edit) (Apache-2.0) focused on deployability.

**Adds:** pure-PyTorch equivalents of the `joyomni_ops` CUDA kernels (bit-exact, max 1 bf16 ulp) with automatic fallback for ROCm/CPU; int4/int8 weight-only Linear plus state-dict loaders; three quantization recipes; a headless WebSocket client.

**Measured (840x480, identical weights/prompt/frames):** bf16 identity 0.314 → **int8 0.359** (15.2 GiB) → attn-int8+MLP-int4 0.199 (11.4 GiB) → int4 0.103 (8.1 GiB). The pipeline is deterministic across restarts (138 dB on identical int4 weights), so these are real. **int4 breaks reference-guided person swapping; int8 preserves it** — the identity signal lives entirely in the attention path.

**Optimization in progress — strip the text stream:** the txt branch (`txt_attn + txt_mlp`) is 8.05 B params, 49.5% of the model. Text K/V depends only on the prompt, so it can be prefilled once per session, cached as static KV, and the weights offloaded to host RAM. The block forward already accepts `skip_text_stream=True` and the pipeline already prefills static KV for the reference image. Expected: **23.9 GB → ~15.8 GB**, enabling 16 GB cards at int8 quality, plus 30-40% faster chunks.

See `NOTICE` for the list of modified files.
