# moe-l2 Supported Models — Measured Summary (updated 2026-08-19)

> All figures are reproducible measurements from the full moe-l2 pipeline (proxy + L2 cache + host-buffer GPU direct compute + selective pin/on-demand pin).
> **2026-08-19: bins-v0.6.0 (per-slot lock build) — all speed figures re-measured on 4090/2080 Ti/5090 with clean output verification.** Replaces the 2026-08-16 bins-v0.5.0 C-scheme figures. V4 still has no valid speed data until a Q4 quant is available.
> Test conditions: 64-128 tokens per request, n_predict=128, c=2048-8192, `GGML_OP_OFFLOAD_MIN_BATCH=1`, `GGML_CUDA_EXPERT_CACHE=1`.
> Links to the detailed per-model reports are at the end of this document.

## Summary table (by model size)

| Model | Parameters | File | Quantization | Experts (active) | GPU VRAM | Host RSS | Speed | Verified on |
|------|------|------|------|------------|----------|----------|------|----------|
| DeepSeek-V2-Lite | 16B MoE | 6 GB | Q2_K | 64 (top-6) | **~10 GB** | **~6.4-6.7 GB** | **139-154 t/s** (4090, v0.6.0) / 86-94 (2080 Ti) / **141-151 (5090, real SM120a)** | RTX 4090 / 2080 Ti / 5090 |
| Qwen3.6-35B-A3B | 32B MoE | 11 GB | UD-IQ2_M | 256 (top-8) | **~6.7-7.2 GB** | **~8.9-11.3 GB** | **25.5-44.2 t/s** (4090, v0.6.0 mixed-domain) / 16.6-28.6 (2080 Ti) / **28-52.5 (5090, real SM120a)** | RTX 4090 / 2080 Ti / 5090 |
| DeepSeek-V4-Flash | **157B MoE** | **85 GB** (3 shards) | UD-IQ2_M | 256 (top-6) | **16.5-16.7 GB** | **17.5-26.8 GB** (on-demand/selective pin) | ⚠️ **N/A — upstream llama.cpp deepseek4 CUDA expert bug** ([#25582](https://github.com/ggml-org/llama.cpp/issues/25582)); UD-IQ2_M/Q4_K_XL garbled on pure vanilla too | RTX 4090 |
| Mixtral-8x7B | 47B MoE | ~16 GB | Q4_K_M | 8 (top-2) | 2.2-2.9 GB | — | 3.7 t/s* | RTX 4090 (cache test conditions) |
| Qwen3-235B-A22B | 235B MoE | 85.7 GB | Q2_K | 128 (top-8) | **13.9 GB** | **54.7 GB** (selective pin) | **~3.9 t/s** (steady state, 4090) | RTX 4090 |

\* Mixtral figures are from the bare llama-server cache-benefit test (experts computed on CPU, not the host-buffer GPU direct-compute main path); reference only.
\*\* 4090/2080 Ti/5090 data measured on the **2026-08-19 bins-v0.6.0 full pipeline** (`moe-l2 start --gpu`, per-domain table switch, default single-table); 5090 is real SM120a data; 3080 Ti still uses the v3.1 multi-arch conditions (bins-v0.3.0).

## Context capability (RTX 3060 12G, 2026-09-09)

Same machine, same day, released engine (A3 expert cache), full-chain `moe-l2 start --gpu`; model Qwen3.6-35B-A3B **UD-IQ2_M** (11.5 GB):

| ctx | VRAM at startup | Gen speed (short prompt) | 45K-token needle retrieval |
|-----|-----------------|--------------------------|----------------------------|
| 65536 | 4966 MiB | — | ✅ |
| 131072 | 6104 MiB | **28.3-29.0 t/s** | ✅ |
| **262144** (model native) | **8700 MiB** | **25.0-25.3 t/s** | ✅ |
| 393216 | 11358 MiB | 22.7-22.9 t/s | ✅ |
| 524288 | ~13.9 GB — does not fit a 12G card | — | — |

- Since PyPI 0.13.1 `--ctx-size` is optional: the CLI targets the model's native context (GGUF `context_length`) and narrows it to what VRAM safely fits — 12G card + IQ2_M auto-lands on 262144 (older builds pinned 8192, a legacy default rather than an engine limit).
- 45K-token needle retrieval passed at every ctx from 64K to 384K with no degradation; long *summarization* degrading on UD-quantized files is a known file-side weak spot (independent of ctx).
- **UD-Q4_K_M** (22.1 GB) on the same card caps at **128K** (13-16 t/s, 7963 MiB): ~1.9 GB more non-expert VRAM, less room for KV.

## VRAM savings (host-buffer expert GPU direct compute + selective/on-demand pin)

| Model | Standard full load | moe-l2 | Savings | Speed retained |
|------|-------------|--------|------|----------|
| DeepSeek-V2-Lite | 23.3 GB VRAM, 65 t/s | **~10 GB, 139-154 t/s** (4090, 2026-08-19 v0.6.0) | **57%** | 205% |
| Qwen3.6-35B-A3B | OOM on 8 GB GPU | **~6.7-7.2 GB, 25.5-44.2 t/s** (4090, 2026-08-19 v0.6.0) | — | ~pre-lazy 46.5 |
| DeepSeek-V4-Flash | OOM on 10-11 GB GPUs | **16.5 GB VRAM / 17.5 GB RSS** (4090；⚠️ 35.96 t/s was UD-IQ2_M 2-bit garbage, no valid speed data) | — | on-demand/selective pin both work |

## Full-pipeline measurements (v3.1 fixed-expert-count eviction, RSS capped)

**Qwen3.6-A3B / DS-V2-Lite (same RTX 3080 10 GB GPU, bare vs full pipeline)**

| Model | Bare server | v3.1 full pipeline | Eviction impact | RSS | VRAM |
|------|-----------|------------|----------|-----|------|
| Qwen3.6-35B-A3B | 8.77 t/s | 9.40 t/s | **+7%** | 4.5 GB | 2.3 GB |
| DS-V2-Lite | 8.66 t/s | 10.21 t/s | **+18%** | 5.6 GB | 3.3 GB |

## Multi-architecture measurements (3 GPUs × 2 models, bins-v0.2.0+, CUDA 12.8)

| GPU | Architecture | DS-V2-Lite | Qwen3.6-A3B | VRAM |
|-----|------|-----------|-------------|------|
| RTX 2080 Ti | sm_75 | 86-94 t/s | 16.6-28.6 t/s | Qwen 5.3 GB / DS 10.0 GB |
| RTX 3080 Ti | sm_86 | 12.25 t/s | 13.28 t/s | 1.1-2.2 GB |
| RTX 5090 | sm_120a | **141-151 t/s** | **28-52.5 t/s** | Qwen 5.6 GB / DS 10.2 GB |
| RTX 4090 | sm_89 | 139-154 t/s | 25.5-44.2 t/s | Qwen 6.7-7.2 GB / DS 10.1 GB |

\* 4090/2080 Ti/5090 measured on the 2026-08-19 bins-v0.6.0 full pipeline (per-domain table switch, default single-table); 5090 is real SM120a data; 3080 Ti measured with the v3.1 multi-arch package (bins-v0.3.0).

## Key conclusions

1. **VRAM is roughly model-bound, not architecture-bound**: DS ~10 GB, Qwen ~9.3 GB on 4090 — comfortable on 10-11 GB cards (old 1.6-4.9 GB figures were P0-garbage inflation, void)
2. **v3.1 eviction does not slow things down**: same-GPU comparison shows the full pipeline is actually faster (Qwen +7%, DS +18% — L2 hot-expert prefetch benefit > eviction overhead)
3. **on-demand pin main path (08-07)**: Qwen 50.2 / DS 37.9 / V4 10.1 t/s (4090) — these 08-07 numbers predate the P0 cache-race and are replaced by the 08-14 fixed-build figures below (Qwen 44-48 / DS 133.2)
4. **selective pin + GPU cache prefetch (08-10, v0.4.0)**: router-table-driven top-K pin — V4 RSS **84.4 → 26.8 GB** (⚠️ 34.67 / 35.96 t/s were UD-IQ2_M 2-bit garbage — no valid speed data, waiting for Q4 quant; the earlier 10.1 t/s was the official unmodified binary); GPU cache prefetch improves cold start +84% (10.7 → 19.7 t/s)
5. **V4 at 0.7-2.2 t/s on the 2080 Ti is the GPU's compute ceiling** (IQ2_M 157B, not an offload cost); no valid 4090 speed figure until a Q4 quant is available
6. **Dynamic pin set (08-09, low-memory mode)**: register only active experts + LRU-evict cold experts — V4 RSS **84 GB → 17-24 GB** (`MOE_L2_LRU_MAX_EXPERTS` 2000≈17 GB / 12000≈24 GB), speed 4-5 t/s (V4 routing is extremely scattered: a 30-turn session touches ~29 GB of distinct experts, and the first touch of a new expert pays a ~2 ms page-fault disk read); Qwen/DS have small working sets and are unaffected. `MOE_L2_PIN_LAYERS=0-2,14-20,36-37` permanently pins the general/sparse layers (~5.4 GB for free). Trade-off: whole-pin is fastest (30.9 t/s) but needs 82 GB RAM; dynamic pin keeps memory bounded but halves V4 speed
7. **Concurrent shared cache (08-12)**: 4-way concurrency sharing the A3 cache / selective pin — Qwen (2080 Ti) total throughput 95.02 (same-domain) / 88.28 (cross-domain) t/s, DS 198.59 / 188.25, V4 (4090) 89.66 / 88.10 — ⚠️ these 08-12 numbers predate the P0 fix and are **P0-void pending re-measure** (2026-08-14 fixed build: DS 4090 4-way same-domain 16/16 clean at ~122 t/s per session, no contention)

## Concurrency (2026-08-12, two machines, three models)

> ⚠️ **All numbers in this table are P0-void** (measured before the 2026-08-13 cache-race fix; garbage-output inflated). 2026-08-14 fixed-build re-measure (DS 4090, 4-way × 4): 16/16 outputs clean, ~122 t/s per session, no contention, no garbage. Table kept for historical reference only.

| Model (GPU) | Single-session t/s | 4-way same-domain throughput | 4-way cross-domain throughput | vs single session | Cross/same |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B (2080 Ti) | 38.4 | **95.02** | **88.28** | 2.3-2.5× | -7% |
| DS-V2-Lite (2080 Ti) | 78.3 | **198.59** | **188.25** | 2.4-2.5× | -5% |
| DeepSeek-V4-Flash (4090, 256 experts) | 35.4-35.8 | **89.66** | **88.10** | 2.5× | -2% |
| Qwen3.6-35B-A3B (4090) | 67.4-68.7 | **174.56** | **144.47** | 2.6× / 2.1× | -17% |
| DS-V2-Lite (4090) | 143.4-143.7 | **369.48** | **354.73** | 2.6× / 2.5× | -4% |

- Conclusion: **concurrency does not break, sharing holds naturally** — under 4-way concurrency (even cross-domain) the active expert sets overlap heavily → no A3 cache contention; total throughput = 2.3-2.5× a single session, so "one AI PC, multiple users at once" is verified in practice
- VRAM: +2.9 GB for 4-way concurrency (all independent KV slots — expected cost, not a leak); RSS +0.2 GB (pages of experts outside the selective pin table do not accumulate)
- ⚠️ Occasional cublas crash on the 4090 at the first cross-domain concurrency (recovers on restart; not consistently reproducible; recorded)
- Full report: [concurrent-cache-sharing-20260812.md](concurrent-cache-sharing-20260812.md)

## Detailed reports

- [DeepSeek-V4-Flash verification report (157B, dual-GPU full pipeline)](deepseek-v4-flash-verify-20260805.md)
- [Qwen3-235B-A22B Q2_K benchmark (235B, three rounds on 4090)](qwen3-235b-a22b-q2k-benchmark.md)
- [Multi-arch three-GPU verification (2080 Ti / 3080 Ti / 5090)](multi-arch-three-gpu-benchmark.md)
- [Qwen3.6-A3B IQ2_M benchmark](qwen3.6-a3b-iq2m-benchmark.md)
- [DeepSeek-V2-Lite Q2_K benchmark](deepseek-v2-lite-q2k-benchmark.md)
- [Cache / sched layer benchmark (DS / Qwen / Mixtral three-model benefit matrix)](cache-sched-layer-benchmark.md)
- [Host-buffer design decisions (CN/EN)](design-decisions.md)
