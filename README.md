# moe-l2

[![CI](https://github.com/yalun753/moe-l2/actions/workflows/ci.yml/badge.svg)](https://github.com/yalun753/moe-l2/actions/workflows/ci.yml)

[English](README.md) | [**中文**](README_zh.md)

**MoE expert offload for low-VRAM GPUs — run 100B+ MoE models (DeepSeek, Qwen, Mixtral) on 10-11 GB cards.** A transparent, OpenAI-compatible proxy that predicts which experts your prompt needs and preloads them into a shared-memory LRU cache, so you can run 16 GB+ MoE models on 10-11 GB GPUs with up to ~60% VRAM savings.

> ⭐ **Found this useful? Give us a star** — it helps others discover the project. [★ Star on GitHub](https://github.com/yalun753/moe-l2)

> 💬 **Measured results & quick start** — full benchmark table, memory numbers, and install guide: [Discussion #2](https://github.com/yalun753/moe-l2/discussions/2)

### Real-world benchmark

| Your GPU | Normally fits | **With moe-l2** | **Measured speed** (RTX 4090) |
|----------|--------------|-----------------|-------------------------------|
| 10-11 GB | — | DeepSeek-V2-Lite (16B MoE) ✅ | **139-154 t/s** |
| **10-11 GB** | 7B dense | **Qwen3.6-A3B (32B MoE) ✅** | **25.5-44.2 t/s** |
| 10-11 GB | — | **DeepSeek-V4-Flash (157B MoE, 85 GB file) ⚠️** | **N/A — upstream llama.cpp deepseek4 CUDA expert bug ([#25582](https://github.com/ggml-org/llama.cpp/issues/25582))** |
| 24 GB | — | **Qwen3-235B-A22B (235B MoE, 85.7 GB file) ✅** | **~3.9 t/s** |

> Speed = RTX 4090 measured (2026-08-19, bins-v0.6.0 per-slot-lock build, full-chain `moe-l2 start --gpu`): DS-V2-Lite 139-154 t/s (single-turn 141, long-context 139), Qwen3.6-A3B 25.5-44.2 t/s (mixed-domain). 2080 Ti full-chain (v0.6.0): Qwen 16.6-28.6 t/s, DS-V2-Lite 86-94 t/s. **5090 (SM120a, v0.6.0): Qwen 28-52.5 t/s, DS 141-151 t/s**. Qwen3-235B-A22B: ~3.9 t/s steady (2026-08-11). See [models-benchmark.md](references/en/models-benchmark.md).

Without moe-l2, a 10-11 GB card **cannot load these models at all** — it OOMs immediately. With moe-l2, a 32B MoE fits in ~9.3 GB VRAM (on-demand pin experts on Qwen3.6-A3B, GPU compute). **DeepSeek-V4-Flash (157B params / 85 GB file, 256 experts, top-6) runs on a 10-11 GB card at 8.3-9.1 GB VRAM** — with selective pin (v4_top100.map), RSS **26.8 GB** (from 84.4 GB whole-pin, −68%); on-demand fallback RSS 17.5 GB (VRAM 16.5-16.7 GB, measured 2026-08-10). ⚠️ **No valid speed data — upstream llama.cpp bug** ([#25582](https://github.com/ggml-org/llama.cpp/issues/25582)): deepseek4 MoE expert layers on CUDA produce garbled output (reproduces on pure vanilla llama-server, no moe-l2 involved); UD-IQ2_M/Q4_K_XL both affected. Waiting for upstream fix. Full report: [deepseek-v4-flash-verify-20260805.md](references/en/deepseek-v4-flash-verify-20260805.md) · **Qwen3-235B-A22B (235B params / 85.7 GB file, 128 experts, top-8) runs on a 24 GB card at ~3.9 t/s steady** — selective pin (top-60/layer, 98.5% coverage): 24 GB VRAM + 55 GB RAM, RSS 80.8 → 54.7 GB (−33%), measured 2026-08-11. Full report: [qwen3-235b-a22b-q2k-benchmark.md](references/en/qwen3-235b-a22b-q2k-benchmark.md) · **All measured models: [models-benchmark.md](references/en/models-benchmark.md)**


### Visual demo (RTX 4090, 2026-08-16)

| Qwen3.6-35B-A3B (32B MoE) — standard vs moe-l2 | DeepSeek-V2-Lite (16B MoE) — 8 GB card vs 24 GB card |
|---|---|
| ![Qwen VRAM comparison](examples/demo-assets/fig1-qwen-vram.png) | ![DS VRAM comparison](examples/demo-assets/fig2-ds-vram.png) |

Summary: **~57-60% less VRAM** — a 10-11 GB card runs what used to need 24 GB (RTX 4090 measured 2026-08-19, bins-v0.6.0: DS 139-154 t/s @ ~10.1 GB / Qwen 25.5-44.2 t/s @ ~6.7-7.2 GB):

![moe-l2 summary](examples/demo-assets/fig3-summary.png)

Live capture (2026-08-16, bins-v0.5.0 C-scheme): Qwen3.6-35B-A3B generating **2,344 tokens with VRAM at ~5.4 GB** (~32 t/s) — watch the VRAM curve stay flat the whole run:

[`examples/demo-assets/demo-vram-animation.mp4`](examples/demo-assets/demo-vram-animation.mp4) (45 s, 1280×720) · raw telemetry: [`examples/demo-assets/rec_data.csv`](examples/demo-assets/rec_data.csv) · full generated text: [`examples/demo-assets/rec_full.txt`](examples/demo-assets/rec_full.txt)

### Benchmarked on RTX 4090 (2026-08-19, v0.6.0)

| Mode | GPU VRAM | Gen speed | What it means |
|------|----------|-----------|---------------|
| Standard (all experts on GPU) | 23.3 GB | 65 t/s | Needs a 24 GB card |
| **moe-l2** (selective pin experts, GPU compute) | **~6.7-10.1 GB** | **DS 139-154 t/s · Qwen 25.5-44.2 t/s** | **Fits in 10-11 GB cards** |
| **Savings** | **~57-60% less** | ~2× of full-GPU speed | Experts stay in CPU RAM, GPU reads them on demand |

> We benchmarked **Qwen3.6-A3B** (32B MoE) and **DeepSeek-V2-Lite** (16B MoE, 64 experts) on RTX 4090 with **selective pin + per-domain table switch + A3 cache** (bins-v0.6.0 per-slot-lock build, 2026-08-19): experts stay in CPU RAM (zero VRAM), the router map pre-selects per-domain hot experts and the scheduler copies only the **activated** experts to GPU each step, hot experts are cached in VRAM. DS-V2-Lite **139-154 t/s** (~10.1 GB VRAM, ~6.4-6.7 GB RSS), Qwen3.6-A3B **25.5-44.2 t/s** (mixed-domain; ~6.7-7.2 GB VRAM, ~8.9-11.3 GB RSS). Full reports: [qwen3.6-a3b-iq2m-benchmark.md](references/en/qwen3.6-a3b-iq2m-benchmark.md) · [deepseek-v2-lite-q2k-benchmark.md](references/en/deepseek-v2-lite-q2k-benchmark.md) · [models-benchmark.md](references/en/models-benchmark.md)

### Selective pin — 低内存模式（V4 RSS 对比，2026-08-10 实测）

![Selective pin RSS comparison — whole-pin 84 GB vs selective pin 26.8 GB vs on-demand 17.5 GB, DeepSeek-V4-Flash UD-IQ2_M on RTX 4090](docs/demo/fig5-selective-pin-rss.png)

*Measured on RTX 4090 (2026-08-10, bins-v0.4.0): whole-pin 84 GB → selective pin 26.8 GB (router-map top-K) → on-demand 17.5 GB. RSS −68% (V4 memory data; speed N/A — upstream deepseek4 CUDA bug [#25582](https://github.com/ggml-org/llama.cpp/issues/25582)). Also: [speed vs RSS scatter](docs/demo/fig5b-selective-pin-speed-rss.png).*

**Selective pin（低内存模式，v0.4.0 起；当前主路径为 v0.6.0 按领域换表）** — a router map (top-K experts per layer, e.g. `v4_top100.map` 43 layers) pre-pins the hot experts as host-pinned; experts outside the map fall back to on-demand pin. No env vars needed for whole-pin default; pass `--router-map <file>` or `--router-top-k N` to `moe-l2 start --gpu`:

```bash
moe-l2 start --model model.gguf --gpu --router-map v4_top100.map
```

### Multi-architecture binaries (bins-v0.8.0, 2026-09-06)

One binary for **all NVIDIA GPUs** — GTX 1080 (sm_61) / P100 (sm_60) through RTX 50-series (sm_120a), plus native Windows (`llama_bins_win.zip`, since bins-v0.8.0). Built with CUDA 12.8; no per-GPU compilation needed. `moe-l2 download-bins` fetches it automatically (platform-aware since PyPI 0.12.0). bins-v0.8.0 extends the v0.7.0 engine with **sm_60 + sm_70** (P100 / V100 / Titan V) and ships a **Windows zip** (llama-server.exe + DLLs, AVX2, CUDA 12.8 runtime bundled, driver ≥ 570). v0.7.0's features carry over: **IQ1_M quantization fix** (Qwen3.8-Flash-Next / Qwen4exp 125B 512-expert models no longer crash in MMQ — routes to MMVQ instead) + **NCCL multi-GPU support restored** (`libnccl.so.2` bundled, `--split-mode layer/row/tensor` works again) + selective pin (router-map driven) + GPU cache prefill + on-demand pin main path + expert-page eviction v3.1 + layered pin + A3 cache 32768 slots + per-domain router table switch (`POST /moe-set-domain`) + per-slot lock optimization + soft_resize / retain-hot-experts v2 (default single-table) + VRAM-adaptive main table top-k + proxy concurrency fixes + flywheel table persistence.

| GPU | Architecture | DS-V2-Lite gen | Qwen3.6-A3B gen | Qwen4exp 125B IQ1_M | VRAM |
|-----|-------------|----------------|-----------------|---------------------|------|
| RTX 2080 Ti | sm_75 (Turing) | 86-94 t/s | 16.6-28.6 t/s | — | Qwen 5.3 GB / DS 10.0 GB |
| RTX 3080 Ti | sm_86 (Ampere) | 12.25 t/s | 13.28 t/s | — | ~1.1-2.2 GB |
| RTX 4090* | sm_89 (Ada) | 139-154 t/s | 25.5-56.3 t/s | **19.8 t/s** | Qwen 6.7-7.2 GB / DS 10.1 GB |
| RTX 5090 | sm_120a | **141-151 t/s** | **28-52.5 t/s** | — | Qwen 5.6 GB / DS 10.2 GB |

\* Rows = full-chain measured (`moe-l2 start --gpu`, per-domain table switch + A3 cache, default single-table). **bins-v0.7.0 (2026-08-28)**: 4090 — Qwen4exp 125B IQ1_M **19.8 t/s** (cache hit 97.3%, 32768 slots), Qwen3.6 **56.3 t/s** (round 3, +17-28% vs v0.6.0's 44-48), DS-V2-Lite **143.5 t/s** (+8%); 2080 Ti — Qwen3.6 **40.3 t/s** (hit 95.1%). Older rows: 4090 v0.6.0 Qwen 25.5-44.2 (mixed-domain) / DS 139-154; 2080 Ti v0.6.0 Qwen 16.6-28.6 / DS 86-94; 5090 v0.6.0 Qwen 28-52.5 / DS 141-151; 3080 Ti v3.1 multi-arch (bins-v0.3.0). All outputs verified clean (no garbage). **Qwen4exp 125B** — 512 experts/layer, IQ1_M: previously crashed (`GGML_ABORT` in MMQ quantize, no support for IQ1_M); fixed by routing to MMVQ + batch-cap split. 3-model 4090 log: `测试数据备份/4090-bins-v070-verify-20260828/`.

### Context capability — 12G card, auto-sized (2026-09-09)

Measured on RTX 3060 12G, Qwen3.6-35B-A3B **UD-IQ2_M** (11.5 GB file), released engine (A3 expert cache), full-chain `moe-l2 start --gpu` — same machine, same day:

| ctx | VRAM at startup | Gen speed (short prompt) | 45K-token needle retrieval |
|-----|-----------------|--------------------------|----------------------------|
| 65536 | 4966 MiB | — | ✅ |
| 131072 | 6104 MiB | **28.3-29.0 t/s** | ✅ |
| **262144** (model native) | **8700 MiB** | **25.0-25.3 t/s** | ✅ |
| 393216 | 11358 MiB | 22.7-22.9 t/s | ✅ |
| 524288 | ~13.9 GB — does not fit a 12G card | — | — |

Since PyPI 0.13.1 `--ctx-size` is optional: the CLI targets the model's native context (`context_length` from the GGUF) and narrows it to what VRAM safely fits, so this model auto-lands on **262144** on a 12G card without any tuning. 45K-token needle retrieval passed at every ctx from 64K to 384K with no degradation (long *summarization* on UD-quantized files is a separate, known weak spot of the file itself — not a ctx limit). UD-**Q4_K_M** (22.1 GB file) on the same card caps at **128K** (13-16 t/s, 7963 MiB): its non-expert layers occupy ~1.9 GB more VRAM, leaving less for KV.

### Which card fits which model (measured, same machine)

| Card | Model / quant | Max ctx | Speed (short prompt) | VRAM at max ctx |
|------|---------------|---------|----------------------|-----------------|
| **12 GB** (RTX 3060) | Qwen3.6-35B-A3B **UD-IQ2_M** (11.5 GB) | **262144** (sweet spot) / **393216** (ceiling) | 25.0-25.3 t/s @256K · 22.7-22.9 t/s @384K · 28.3-29.0 t/s @128K | 8700 MiB @256K · 11358 MiB @384K |
| **12 GB** | Qwen3.6-35B-A3B **UD-Q4_K_M** (22.1 GB) | **131072** | 13-16 t/s | 7963 MiB |
| **8 GB** | Qwen3.6-A3B class (IQ2) | depends on quant — start lower | ~11-17 t/s (expert-cache hit rate 65-76%) | — |

Since PyPI 0.13.1 context is auto-sized: leave `--ctx-size` out and the CLI targets the model's native context, narrowed to what the VRAM budget safely fits (no OOM). Quantization matters as much as ctx: on a 12 GB card the IQ2-class file reaches 256K while Q4 caps at 128K — the difference is non-expert weights competing with the KV budget.

### Long documents — first read vs repeat read (RTX 3060 12G, 2026-09-09)

| Scenario | Wall time (45K-token document) |
|----------|-------------------------------|
| **First** read (cold prefill, ≈550 tok/s prefill) | 80-200 s |
| **Repeat** read (prompt cache hit) | 85-96 s of generation, prefill ≈ free |

45K-token needle retrieval passed at every ctx from 64K to 384K, and quality does not degrade on repeated reads — so if you ask several follow-up questions about one long document, keep them in the same session: the document is prefilled once and reused. Starting a fresh session for each question re-pays the prefill.

### Concurrent requests — shared cache, no speed loss (2026-08-12)

4 parallel slots share one A3 expert cache / selective-pin table — verified on 2080 Ti and 4090 with **Qwen3.6-35B-A3B**, **DS-V2-Lite** and **DeepSeek-V4-Flash (256 experts, spread routing)**:

| Model (GPU) | Single session | 4× concurrent, same domain | 4× concurrent, cross-domain | vs single |
|---|---|---|---|---|
| Qwen3.6-35B-A3B (2080 Ti) | 38.4 t/s | **95.02** total (23.76×4) | **88.28** total (21.7-22.2×4) | 2.3-2.5× |
| DS-V2-Lite (2080 Ti) | 78.3 t/s | **198.59** total | **188.25** total | 2.4-2.5× |
| DeepSeek-V4-Flash (4090) ⚠️ | 35.4-35.8 t/s | **89.66** total | **88.10** total | 2.5× |

> ⚠️ V4 行速度 N/A — 上游 deepseek4 CUDA 专家路径 bug（#25582）。

Concurrent throughput = **2.3-2.5× a single session**; cross-domain vs same-domain is only **-5-7%** — no per-domain cache pools needed. VRAM grows only by the per-slot KV cache (+2.9 GB for 4 slots), RAM stays flat (+0.2 GB). **One AI PC can serve multiple users at once.** Full report: [concurrent-cache-sharing-20260812.md](references/en/concurrent-cache-sharing-20260812.md)

## Quick start

**One-line install (Linux x86_64 + NVIDIA GPU):**

```bash
curl -fsSL https://raw.githubusercontent.com/yalun753/moe-l2/main/scripts/install.sh | bash
```

The installer checks your GPU/driver/Python, installs moe-l2 from PyPI,
downloads the pre-built CUDA binaries, optionally downloads a demo model
(Qwen3.6-35B-A3B, ~11.5 GB, resumable), then runs a self-check.

**Manual install:**

```bash
pip install moe-l2                   # keyword-only predictor (zero extra deps)
pip install moe-l2[predictor]        # hybrid: keyword + semantic embedding
moe-l2 download-bins                 # pre-built CUDA llama-server (on-demand pin patched)
moe-l2 model download --model qwen3.6-35b   # optional demo model (~11.5 GB)
moe-l2 start --model model.gguf --gpu
```

Useful commands:

```bash
moe-l2 doctor                        # environment self-check (GPU/CUDA/Python/disk)
moe-l2 model list                    # list downloadable models
moe-l2 model download --model <name> # download model (resumable, via hf-mirror)
```

Your tools (curl, Open WebUI, LangChain) connect to `localhost:11435` — no client changes needed.

## How it works

MoE models have many "experts" but only activate a few per token. moe-l2 predicts your prompt's domain (codegen, math, chinese_tech, etc.) and preloads the relevant experts into an mmap'd LRU cache before they're needed.

```
user → moe-l2 proxy (localhost:11435)
    ├── predict domain (keyword → TF-IDF → semantic)
    ├── on-demand pin experts (lazy mmap + register, zero VRAM)
    ├── hot experts cached in VRAM (A3 LRU)
    └── forward to llama-server (localhost:11436, CUDA GPU)
        └── GPU reads pinned experts via PCIe DMA; cold pages evicted
```

## Usage

### 1. L2 proxy with GPU on-demand pin (recommended)

Start the transparent proxy with the bundled on-demand pin llama-server:

```bash
moe-l2 start --model /models/DeepSeek-V2-Lite.Q4_K_M.gguf --gpu
```

The proxy exposes OpenAI-compatible endpoints — all your tools work through it (curl, open-webui, langchain):

```bash
# streaming
curl http://localhost:11435/v1/chat/completions -d '{
  "model":"qwen3:4b",
  "messages":[{"role":"user","content":"write a Python script"}],
  "stream":true
}'

# blocking
curl http://localhost:11435/v1/chat/completions -d '{
  "model":"qwen3:4b",
  "messages":[{"role":"user","content":"hello"}],
  "stream":false
}'
```

### 2. Monitor cache stats

```bash
moe-l2 stats --port 11435
```

Example output:
```
moe-l2 cache stats
  requests:     47
  hits:         42     (89.4%)
  misses:        5
  slots_used:  32/48  (66.7%)
  memory:     456 MB  (68.3% of 668 MB)
```

### 3. Use as a library

```python
from moe_l2 import predict, predict_hybrid, domain_to_expert_ids
from moe_l2.cache import L2Cache

# Predict domain (zero-dependency mode)
domain = predict("print hello world")  # → "codegen"

# Or use the hybrid semantic predictor
domain = predict_hybrid("how do I sort a list?")  # → "codegen"

# Preload experts
cache = L2Cache(model_path="model.gguf", l2_size="4GB")
cache.preload(domain_to_expert_ids[domain])
```

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                       HTTP client                          │
│     curl / open-webui / langchain / any OpenAI client      │
└──────────┬─────────────────────────────────────────────────┘
           │ POST /api/chat
           ▼
┌────────────────────────────────────────────────────────────┐
│               moe-l2 Proxy (port 11435)                     │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Domain Predictor                                    │   │
│  │  - Keyword mode: zero deps, instant classification   │   │
│  │  - Hybrid mode: +sentence-transformers for context   │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        │ domain                             │
│  ┌─────────────────────▼───────────────────────────────┐   │
│  │  L2 Cache (mmap'd shared memory)                    │   │
│  │  - LRU eviction policy                              │   │
│  │  - Async preload: next-prediction prefetch          │   │
│  │  - Thread-safe concurrent access                    │   │
│  │  - Zero-copy mmap from SSD → RAM                    │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        │ forward request                    │
└────────────────────────┼───────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────────┐
│         llama-server (port 11436, CUDA GPU)                │
│         on-demand pin experts: lazy mmap, zero VRAM       │
│         GPU reads pinned experts via PCIe DMA             │
│         hot experts cached in VRAM (A3 LRU, 2048 slots)   │
│         cold expert pages evicted (v3.1, RSS capped)      │
└────────────────────────────────────────────────────────────┘
```

## CLI reference

| Command | Description |
|---------|-------------|
| `moe-l2 start --model <path> --gpu` | Start proxy + on-demand pin llama-server (recommended) |
| `moe-l2 start --model <path> --l2-size <size>` | Start proxy + cache only (no GPU) |
| `moe-l2 stats --port <port>` | Show live cache stats |
| `moe-l2 download-bins [--release TAG]` | Download pre-built GPU binaries from GitHub |
| `moe-l2 collect --model <path>` | Collect MoE routing data → `~/.moe-l2/maps/domain_expert_map.json` |
| `moe-l2 stop --port <port>` | Stop proxy |

Options:
- `--model auto`: scan `/opt/data/models/*.gguf`
- `--l2-size 4GB` / `--l2-size 512MB`: target cache size (proxy-only mode)
- `--port 11435` (default)
- `--gpu`: enable GPU mode (requires CUDA + NVIDIA GPU; spawns bundled on-demand pin llama-server on 11436)

> **GPU binaries**: Not tracked in git (bundled as `llama_bins.tar.gz` on Linux / `llama_bins_win.zip` on Windows, multi-architecture — sm_60/61/70/75/86/89/120a since bins-v0.8.0, one binary for all NVIDIA GPUs incl. P100/V100, ships cuda-libs/CUDA runtime DLLs). Fetched at runtime via `moe-l2 download-bins` (platform-aware since PyPI 0.12.0). When you `pip install moe-l2`, binaries are included. For git-clone users, run `moe-l2 download-bins` to fetch them from GitHub Release.

## Platform requirements

- **Linux x86_64 + NVIDIA GPU** — primary platform (CUDA `.so` + `llama-server` bundle)
- **Windows 10/11 x64 + NVIDIA GPU** — native support since bins-v0.8.0 (`llama_bins_win.zip`, llama-server.exe + DLLs; same A3 expert cache / router-map features as Linux)
- macOS, and ARM Linux are **not supported**
- **NVMe SSD strongly recommended**
- NVIDIA GPU required for `--gpu` mode

### Windows (native, bins-v0.8.0+)

Install moe-l2 normally, then fetch the Windows engine — `download-bins` auto-picks `llama_bins_win.zip` on Windows:

```
pip install moe-l2
moe-l2 download-bins
moe-l2 start --model C:\models\Qwen3.6-35B-A3B-UD-IQ2_M.gguf --gpu
```

The zip ships `llama-server.exe` + `llama-cli.exe` + all DLLs (multi-arch cubins: sm_60/61/70/75/86/89/120a, AVX2, CUDA 12.8 runtime DLLs included — no CUDA Toolkit install needed, NVIDIA driver ≥ 570 required). To generate a per-domain router map on Windows, run `moe-l2 collect --model <your.gguf>` (uses the bundled llama-cli.exe automatically). Verified on RTX 3060 12 GB (Qwen3.6-35B-A3B ~21-22 t/s LRU hot-cache, same engine as Linux).

## More data

| Metric | Standard | With moe-l2 |
|--------|----------|-------------|
| Prompt processing (DS-V2-Lite) | 110 t/s | 99 t/s · **308 t/s** (sched-cache=0.25) |
| Generation speed (DS-V2-Lite) | 65 t/s | 139-154 t/s · 39.2 t/s (sched-cache=0.25, 08-02) |
| Generation speed (Qwen3.6-A3B) | — | 25.5-44.2 t/s |
| VRAM used (DS-V2-Lite) | 23.3 GB | **~10 GB** |

The speed tradeoff is intentional and small: expert weights live in CPU RAM (lazy mmap, zero VRAM) and are pinned on first touch — the GPU reads them directly via PCIe DMA, hot experts are cached in VRAM (A3 LRU), and cold pages are evicted to keep RSS capped. On the 2026-08-19 bins-v0.6.0 build, DS-V2-Lite reaches 139-154 t/s gen at ~10.1 GB VRAM — faster than full-GPU at ~43% of the VRAM.

## Expert offload & cache fast path (llama.cpp)

Beyond the proxy layer, moe-l2 ships llama.cpp patches that compile expert handling directly into the CUDA backend — no proxy needed. Two mechanisms:

**1. Selective pin expert GPU fast path (2026-08-10, current main path).** Expert tensors live in CPU RAM via lazy mmap (zero VRAM). A router map (top-K experts per layer) pre-pins the hot experts as host-pinned (`cudaHostRegister`), so the GPU reads them directly via PCIe DMA; experts outside the map fall back to on-demand pin. Hot experts are cached in VRAM (A3 LRU, 32768 slots max) and cold pages are evicted (v3.1) to keep RSS capped. Measured (2026-08-19, bins-v0.6.0): DS 139-154 / Qwen 25.5-44.2 t/s (4090, mixed-domain). V4: ⚠️ speed N/A — upstream deepseek4 CUDA expert bug (#25582).

**2. A3 LRU expert cache (historical, `--expert-cache`).** An LRU cache that keeps recent experts on GPU. In the old `--cpu-moe` CPU-compute architecture it cut VRAM from 6.6 GB → 1.2 GB (5.64×) at 8.2 t/s. In the current on-demand pin architecture the cache is hooked into the scheduler copy layer (`GGML_CUDA_EXPERT_CACHE`) and only pays off for small, frequently-hit experts (see below).

### When the cache helps (and when it doesn't)

The sched-cache only pays off when experts are **small and frequently hit**. Verified on RTX 4090 (host-buffer, 2026-08-02):

| Model | Expert size | Top-k | Cache value |
|-------|------------|-------|-------------|
| DS-V2-Lite | 1.55 MB | top-6 | ✅ **Prompt +211%, Gen +5%** (cache=0.25) |
| Qwen3.6-A3B | ~1 MB | top-8 | ❌ no gain (experts too small, copy cost already trivial) |
| Mixtral-8x7B | 252 MB | top-2 | ❌ no gain, +660 MiB VRAM (top-2 hit rate too low) |

Key findings (2026-08-02, cache hooked into the scheduler input-copy layer):

- The cache sits in `copy_experts`: on hit it does a D2D copy (no PCIe round-trip), on miss it falls back to the pinned-host CPU→GPU path and writes back. It only intercepts single-expert groups.
- Benefit = **expert size × hit rate**. DS (1.55 MB, top-6) wins big; Qwen (~1 MB) pays for itself at best; Mixtral (252 MB, top-2) never hits enough to pay for its VRAM slots.
- Recommended: `GGML_CUDA_EXPERT_CACHE=0.25` for DS-class models (16 slots/layer cover all hot experts, VRAM unchanged). Leave it off for Qwen/Mixtral.

> Run the demo yourself: `bash examples/demo_a3_compression.sh` (edit paths first).

## Related work

### AirLLM (lyogavin/airllm, ~29k stars)

AirLLM is a general-purpose layer-offload scheme for very large models. Its scheduling granularity is the **full Transformer layer**: during inference only one layer's weights stay in VRAM, everything else is swapped to/from disk — giving an extreme low-VRAM floor (4GB GPU runs 70B). But it has three weaknesses: ① every generated token requires reading/writing a full layer to/from disk, so IO cost is huge and interactive speed is very low; ② no MoE-specific routing prediction or expert hot cache (per-expert streaming only started in 2026-07, with Kimi K3), so repeated prompts keep triggering heavy disk reads; ③ built on native Hugging Face Transformers, with no OpenAI-compatible serving interface out of the box, making it awkward to wire into Open WebUI, LangChain, etc.

| Dimension | AirLLM | moe-l2 |
|-----------|--------|--------|
| Scheduling unit | Full Transformer layer | **Per-expert (sparse-optimal)** |
| Target models | All models (dense + MoE) | **MoE-optimized (DeepSeek / Qwen / Mixtral)** |
| Weight format | Native Hugging Face weights | **GGUF (llama.cpp ecosystem)** |
| Platform | Windows / macOS / Linux, incl. CPU | Linux x86_64 + NVIDIA GPU |
| MoE memory | Whole-layer disk swap, no hot cache | **85GB V4: 8.3GB VRAM + 11-12GB RSS cap (measured)** |
| MoE speed | Per-layer disk thrash, batch-offline only | Hot-expert cache cuts disk IO, real-time chat (Qwen full-chain 9.3 t/s measured) |
| Serving API | Python-code only, no web service | **Built-in OpenAI-compatible proxy (:11435), drop-in** |
| GPU support | Native transformers, manual CUDA setup | **Multi-arch kernels via download-bins, GTX10xx–RTX50xx** |
| Multi-shard GGUF | No specific support | **Fixed multi-shard metadata parsing, 85GB 3-shard V4 stable** |

**Which to choose**: pick moe-l2 if you run MoE models (DeepSeek/Qwen) locally for chat, have an 8–12GB older NVIDIA card, want an OpenAI API for tooling, or use multi-shard giant GGUFs. Pick AirLLM if you need dense (non-MoE) models, use Windows/macOS/AMD or CPU-only environments (moe-l2 currently requires Linux + NVIDIA), only do one-shot batch generation, or must stay with native HF weights.

## Known issues & tips

- **Long-form summarization at `temperature ≥ 1.0`** (reproducible, measured 2026-09-12 on RTX 3060 12G): with a 14.5K-token document and `max_tokens 2600`, the model sometimes spends the whole budget thinking and emits an empty body — **UD-Q4_K_M 3/6 rounds, a non-UD Q4_K_M 2/6 rounds** (same machine, same engine, same fixture; the two are statistically indistinguishable at n=6). Needle retrieval is unaffected: at `temperature 0.7` it passes at 64K-384K on the same files. **Tip: keep `temperature` at 0.7-0.8 for long inputs** (every example in this README uses 0.7); if you get an empty body, retry or lower the temperature.
- **L0 zero bytes are not a corrupted-file signature**: `blk.0` expert weights show ~14% zero bytes in *both* UD and non-UD Q4_K_M files (13.8% / 15.5%) and 32-36% in IQ2-class files, while every other layer sits at 1-2%. It scales with quantization width, not with the conversion toolchain — an earlier "UD toolchain dropped data" reading was wrong (verified 2026-09-12 by byte-level inspection of a second, independently produced file). The same long-input behaviour appears on vanilla `llama-server` with no moe-l2 involved, so it is a model/quantization property, not an engine bug.
- **Give long documents one warm-up**: the first read of a 45K-token document costs the prefill (80-200 s on a 3060); follow-up questions in the same session reuse it.
- **Windows native (bins-v0.8.0+)**: console encoding is handled since PyPI 0.13.1 (previously `moe-l2 start` could crash on a GBK console when printing symbols). CUDA runtime requires driver ≥ 570.

## Testing

Automated CI runs on every push (GitHub Actions, Python 3.10–3.13): `ruff` lint, `pytest` with coverage (fail below 50%), and package build. Status badge: [![CI](https://github.com/yalun753/moe-l2/actions/workflows/ci.yml/badge.svg)](https://github.com/yalun753/moe-l2/actions/workflows/ci.yml)

- **113 tests** covering the Python scheduler core: domain predictor (keyword boundaries, fallback), L2 cache (LRU eviction, pinning, domain switching), GGUF weight reader (synthetic models), transparent proxy (live fake-backend HTTP, blocking + SSE), CLI helpers and the training data flywheel.
- **Coverage**: 72–88% on the core modules (cache 88%, proxy 78%, gguf_reader 73%, predictor 72%), ~55% total.
- Run locally:
  ```bash
  uv sync --group dev
  uv run pytest tests/
  uv run ruff check moe_l2/ tests/
  ```

> The C++ side (llama.cpp on-demand-pin / expert-cache patches) is GPU-bound and is verified by the end-to-end benchmark reports in `references/` — see [models-benchmark.md](references/en/models-benchmark.md).

## Project status

- ✅ Domain predictor (keyword + optional semantic)
- ✅ L2 cache (mmap LRU, thread-safe, async preload)
- ✅ Transparent proxy (HTTP/SSE forwarding)
- ✅ CLI with auto model detection, GPU mode, and `collect` (routing data → expert map)
- ✅ **Selective pin + GPU prefill (2026-08-10, v0.4.0, current main path)**: router-map-driven top-K pin → V4 RSS **84.4 → 26.8 GB** (on-demand fallback 17.5 GB; ⚠️ V4 speed N/A — upstream deepseek4 CUDA bug #25582); DS **139-154** / Qwen **25.5-44.2** t/s on 4090 (2026-08-19, bins-v0.6.0); GPU cache prefill lifts cold-start round1 10.7 → 19.7 t/s (+84%). (Prior milestones: host-buffer fast path 08-02 → on-demand pin 08-07 → selective pin 08-10.)
- ✅ **bins-v0.6.0 (2026-08-19)**: cache slot cap 16384→32768; per-slot lock optimization (concurrent correctness restored — v0.5.0 lock-free DS concurrent was garbage; same-machine A/B 4090: Qwen 31.76 vs 32.42 ≈parity, **DS 140.43 vs 113.15 +24%**); retain-hot-experts v2 (default single-table, `MOE_L2_POOL_SIZE=3` opt-in); VRAM-adaptive main top-k (≤12G=75 / >12G=100); proxy concurrency fixes (DS long 500 gone). Three-card regression (2080Ti/4090/5090) all outputs clean.
- ✅ Expert cache boundary verified on Mixtral 8x7B / RTX 4090 (2026-08-02, sched-cache): cache benefit = expert size × hit rate — DS-V2-Lite (1.55 MB, top-6) gets Prompt +211% / Gen +5% at cache=0.25; Qwen (~1 MB) and Mixtral (252 MB, top-2) get no gain. Recommended: cache=0.25 for DS-class, off otherwise.
- ✅ **DeepSeek-V4-Flash (157B MoE) verified (2026-08-05)**: 85 GB 3-shard GGUF runs on 2080 Ti (11 GB) — VRAM 8.3-9.1 GB, RSS capped by expert-page eviction v3.1 (fixed-expert-count LRU, `MOE_L2_LRU_MAX_EXPERTS`), multi-shard GGUF parsing fix shipped. [Full report](references/en/deepseek-v4-flash-verify-20260805.md)
- ✅ PyPI package (`moe-l2`)

---

## License

**Apache 2.0.** See [LICENSE](LICENSE) for details.
