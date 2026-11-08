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

### 1. 剥离文本流权重（已实现并实测）

模型每层的参数构成：

| 分支 | 每 block | ×40 | 占比 |
|---|---:|---:|---:|
| 图像流 `img_attn + img_mlp` | 201 M | 8.05 B | 49.5% |
| **文本流 `txt_attn + txt_mlp`** | **201 M** | **8.05 B** | **49.5%** |

做法（三个开关，默认全关，可叠加）：

1. `JOYOMNI_TEXT_KV=1`：会话开始时按 **denoise step** 各跑一次文本分支，逐层缓存 `txt_k`（qk-norm 之后）与 `txt_v`；
   之后每个 chunk 传 `skip_text_stream=True`，attention 的 K/V = 静态缓存 + 当前图像 + 文本。
2. `JOYOMNI_TEXT_OFFLOAD=1`：K/V 抓完后把文本流权重（fp8 孪生）搬回主机内存，需要时逐 block 换入。
3. `JOYOMNI_TIGHT_VRAM=1`：再关掉 CUDA graph、每 chunk 释放一次 allocator。

实测（RTX 5090 / 840×480 / `1101_face++` / 41 帧）：

| 配置 | 稳态显存 | 首帧 | chunk |
|---|---:|---:|---:|
| 基线 | 25.5 GiB（26124 MiB） | 1.8 s | 0.314 s |
| + 文本 K/V + 权重卸载 | 19.5 GiB（19926 MiB） | 6.4 s | 0.280 s |
| + tight（关 graph） | **15.3 GiB**（15668 MiB） | 10.7 s（首次 prompt）/ **1.4 s**（缓存命中） | 0.314 s |
| + tight + `JOYOMNI_KV_DTYPE=int8` | **14.7 GiB** | 10.7 s | 0.316 s |

> 注意：文本 K/V 缓存**本身不省显存**——启动时 24.8 GiB vs 基线 23.6 GiB（缓存 0.46 + graph 池 0.46），
> 必须和 `JOYOMNI_TEXT_OFFLOAD` 一起用。表中数字为"跑完一个会话"后的 nvidia-smi，括号内为原始 MiB。
> 文本 K/V 抓取的结果按（prompt embeds + 参考图 latent + steps + seed）做主机侧缓存（LRU 2），同一组输入的新会话直接拷回显存：首帧 **11.1 → 1.4 s**，逐帧与首次抓取 **138.13 dB** 完全一致（不同 prompt 各自 8–11 s 首次抓取，LRU 满时淘汰最旧一条）。

- **16 GB 卡可跑**：`JOYOMNI_VRAM_CAP_GB=16` 的硬上限下 17 帧会话跑通（reserved 14.5 GiB，峰值 13.7 GiB）。
- dit 每 chunk 0.219 → 0.185 s（graph 稳态，文本分支不再参与）。
- **代价**：缓存的文本 K/V 与完整文本流前向不是逐比特等价——PSNR 21.0 dB（与 int8 量化的 21.8 dB 同量级），
  身份相似度持平（0.67–0.69）；卸载与 tight 本身不改数值（逐帧 138 dB 一致）。
- 可选 `JOYOMNI_KV_DTYPE=int8`：缓存的图像 K/V 存 int8（按 token/head 缩放），KV 1.44 → 0.74 GiB，稳态 15.3 → **14.7 GiB**；代价是与 bf16 缓存的输出差 **26.8 dB**（可复现测量），只在显存最紧时用。
- 会话可复现：`seed_everything(seed)` 之后，同 prompt/seed 的连续两个会话逐帧一致（138 dB）；改动前因为参考图 VAE 采样走全局 RNG，两个会话相差约 25 dB。

### 2. 其他方向（本轮实测）

- **参考图 KV 复用** ✅ 已实现：参考图 prefill 跑的是 `skip_text_stream=True`，K/V 只依赖参考图 latent。
  现在在主机侧留一个 LRU（2 条），同一参考图的新会话直接 memcpy 回显存，省一次 40 个 block 的前向（≈0.4 s/会话），
  数值完全一致（复用 vs 重算逐帧 138.13 dB）。
- **VAE 分时驻留** ❌ 不适用：单卡部署里 encode / decode / pseudo-encode 三个角色共享同一份 VAE
  （`_vae_for_role` 走 share 分支，`decode_vae = pipeline.vae`），全程只有 1.43 GiB 一份，没有"另一份"可释放。
  原先说的省 0.7 GB 基于"两份 VAE"的假设，实测不成立；只有在 `--vae-decode-device` 指定第二张卡时才会出现第二份。
- **HEAD 输出量化** ⚠️ 已实现但**不建议开**：`JOYOMNI_FP8_HEAD=1` 把 `proj_out` + `condition_embedder`
  （time_proj / time_embedder / text_embedder，152 M，291 MiB bf16）转成 FP8，省 145 MiB；
  但与 bf16 输出差 **23.2 dB**（可复现测量），与整模型 8-bit 量化的 21.8 dB 同量级 —— 调制向量太敏感。
- **MLP 5-bit**：注意力 int8 + MLP 5-bit 约 13.3 GiB，误差预计 0.35%，相似度有望到 0.27–0.29（未做）

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
