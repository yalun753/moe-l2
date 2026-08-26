# Changelog

All notable changes to moe-l2 are documented here. PyPI releases track the Python package (`moe-l2`); GitHub Releases track the pre-built CUDA binaries (`bins-vX.Y.Z`, fetched via `moe-l2 download-bins`).

Format: Keep a Changelog 1.1 style — Added / Changed / Fixed.

---

## [0.10.1] - 2026-08-26

### Fixed
- **AVX512 compatibility (bins-v0.6.1)** — v0.6.0 binaries were compiled with AVX512 instructions and crashed on startup with SIGILL (exit 132) on CPUs without AVX512 (e.g. Intel desktop Alder Lake 12th-gen). bins-v0.6.1 is rebuilt with AVX2 only (`GGML_AVX512=OFF`), compatible with all x86_64 CPUs (AVX2 exists on every 2013+ CPU).

### Changed
- **CUDA 12.8 multi-arch rebuild (bins-v0.6.1 asset refreshed)** — the previous binary shipped with CUDA 11 runtime (compiled against a PATH-shadowed CUDA 11.5 nvcc), which ran on 30-series cards via a non-native path. Rebuilt with CUDA 12.8 toolchain + `CMAKE_CUDA_ARCHITECTURES="61;75;86;89;120a"`; verified on 3060 (sm_86) and 2080 Ti (sm_75). **~2× faster on 3060** (see Verified).
- `_DEFAULT_BINS_TAG` → `bins-v0.6.1` — multi-arch (sm_61/75/86/89/120a) AVX2 + CUDA 12.8 build; fixes SIGILL on no-AVX512 CPUs while keeping full GPU family support. **Note: CUDA 12 runtime requires NVIDIA driver ≥ 570** (older drivers must upgrade).
- **Verified 2026-08-26** (Qwen3.6-35B-A3B UD-IQ2_M, thinking disabled):
  - RTX 3060 12G (sm_86, Win10 + WSL2, no-AVX512 CPU): **23.6 t/s steady-state** (vs 15.5 t/s on CUDA 11 build, +52%), 5.5-5.7 GB VRAM, no SIGILL, full-chain OK — first-ever verified Windows/WSL2 run of moe-l2
  - RTX 2080 Ti (sm_75, native Linux): 22.3 t/s, VRAM peak 7.2 GB (expert cache active)

---

## [0.10.0] - 2026-08-22

### Added
- **Per-slot lock optimization (C++ expert-cache)** — replaces the global `std::mutex` with per-slot spinlocks: lock-free prescan get + single-slot lock (re-validate + atomic clock LRU update); locked-victim set with double-check (≤16 re-selects); resize/soft_resize/free lock all slots. Removes global-lock serialization (get scans thousands of slots per token) and **restores concurrent correctness** — v0.5.0 shipped lock-free and DS concurrent output was guaranteed garbage. Same-machine A/B (4090): Qwen 31.76 vs unlocked 32.42 t/s (≈parity, noise), **DS 140.43 vs unlocked 113.15 t/s (+24%)**.
- **`soft_resize` + retain-hot-experts v2** — domain switch grows the cache instead of clearing it; old-domain hot experts stay resident (switch-back first request +30% measured on 2080Ti Qwen). Default single-table after 4090 measurement (retain pool cost ~20% on DS in mixed alternating domains); opt-in via `MOE_L2_POOL_SIZE=3` / `MOE_L2_RETAIN_TOP_K`.
- **VRAM-adaptive main table top-k** (`vram_adaptive.py`) — ≤12G cards use top-75, >12G use top-100 automatically.
- **Cache slot cap doubled** — `EXPERT_CACHE_MAX_SLOTS` 16384 → 32768.
- `domain_router_flywheel.py` `DEFAULT_TOP_K` 75 → 100.

### Fixed
- **Proxy concurrency deadlock + DS long-ctx HTTP 500** — domain switch was a synchronous per-request `POST /moe-set-domain` (10s timeout) and `maybe_retrain` re-trained synchronously (TF-IDF + LinearSVC, seconds); both now run throttled (30s same-domain window) / in background threads.
- v0.5.0 lock-free expert cache concurrent garbage (DS) — fixed by per-slot locks above.

### Changed
- `_DEFAULT_BINS_TAG` → `bins-v0.6.0` — multi-arch (sm_61/75/86/89/120a, CUDA 12.8, no NCCL), includes per-slot locks + soft_resize + MAX_SLOTS 32768 + proxy concurrency fixes.
- README (EN/ZH): speed tables updated to v0.6.0 full-chain (2026-08-19): 4090 DS 139-154 / Qwen 25.5-44.2 t/s; 2080 Ti DS 86-94 / Qwen 16.6-28.6 t/s; 5090 DS 141-151 / Qwen 28-52.5 t/s. All outputs verified clean on all three cards.
- Default retain-pool off (single-table), matching v0.5.0 C-scheme behaviour.

---

## [0.9.0] - 2026-08-16

### Added
- **C-scheme per-domain table switch** — per-domain router maps with `POST /moe-set-domain`: switching domains swaps the active expert table instead of reusing one global table.
- **Retain-hot-experts v1 (retain pool)** — domain switch keeps previous-domain hot experts resident instead of clearing the cache (v2 landed in 0.10.0 with `soft_resize`).
- **DS crash fix** — cache slot resize on table switch no longer crashes concurrent DeepSeek inference.
- **Flywheel table persistence (A-fix)** — learned router map survives restarts.
- `_DEFAULT_BINS_TAG` → `bins-v0.5.0` — multi-arch (sm_61/75/86/89/120a, CUDA 12.8), includes selective pin + C-scheme table switch + GPU cache prefill + on-demand pin + eviction v3.1 + layered pin + A3 cache 2048 slots + P0 fix.

### Verified (2026-08-16, 2080 Ti / 4090 / 5090, clean outputs)
- DS-V2-Lite: 83-88 / 127-137 / 145-153 t/s
- Qwen3.6-A3B: 13-25 / 20-34 / 45-51 t/s

### Changed
- README (EN/ZH): speed tables updated to bins-v0.5.0 three-card data.

---

## [0.8.1] - 2026-08-14

### Fixed
- **Flywheel router-map per-model isolation (flywheel B)** — each model now converges its own dynamic table (`domain_router_map_flywheel_{model_id}.json`); models no longer pollute each other's hot-expert statistics. Legacy single-file `domain_router_map_flywheel.json` retired (kept as read-only fallback when no model_id is available).
- **Domain hotness ranking** — flywheel tracks per-domain request frequency (`_dom_freq`) as the basis for hot-domain routing decisions.
- `_DEFAULT_BINS_TAG` → `bins-v0.4.1` — includes the **P0 fix** (expert-cache D2D copy now carries `es+padding`; concurrent cache-set protected by a mutex). Cache-hit reads no longer return garbage from padding bytes; verified full-chain on 2080 Ti & 4090.

### Changed
- README (EN/ZH): download/install docs point to `bins-v0.4.1` (~1.6 GB multi-arch, sm_61/75/86/89/120a).

---

## [0.8.0] - 2026-08-10

### Added
- **Selective pin (router-map driven)** — opt-in via `MOE_L2_ROUTER_FILE=<router-map>`: pin only the top-K experts per layer from the routing table instead of the whole tensor; experts outside the table fall back to on-demand pin. Measured on RTX 4090 / DeepSeek-V4-Flash (85 GB): RSS **84.4 → 10.4 GB (↓88%)** with **30.9 t/s zero regression** (vs 30.2 whole-pin). Startup 10 s vs 40 s whole-pin.
- **GPU cache prefill (phase 2)** — preload the top-K experts into the A3 GPU cache at load time: cold-start round1 **10.7 → 19.7 t/s (+84%)**, steady state unchanged (~30-31 t/s).
- **Flywheel router-map auto-learning** — `moe_l2/domain_router_flywheel.py` aggregates real inference routing (EXPERT log lines), rebuilds the router map at thresholds (5000 records), new domains added automatically; `load_mapping()` prefers the flywheel table, falls back to the static table.
- **`--router-map` CLI option** + `MOE_L2_ROUTER_FILE` env var; V4 43-layer top-100 map bundled (`moe_l2/data/domain_router_map_v4.json`, `domain_router_map_v4_topics.json`, Qwen `domain_router_map_qwen.json`).
- bins-v0.4.0 multi-arch binaries (sm_61/75/86/89/120a, CUDA 12.8, no NCCL) — includes selective pin + prefill.
- **Corrected V4 baseline**: 30.9 t/s (the earlier 10.1 t/s was the **vanilla llama.cpp binary**; the moe-l2 optimized build was always ~30 t/s). 2080 Ti full-chain re-measured (2026-08-10): Qwen **47.24** / DS-V2-Lite **87.25** t/s (+200~700% vs vanilla).

### Changed
- Default behavior unchanged (whole-pin); selective pin is opt-in via router file.

### Fixed
- V4 startup timeout: `_wait_for_llama_server` 30 → 180 s (V4 85 GB loads in ~90 s; the old 30 s always TIMED OUT and killed the server)
- `maybe_init` GPU cache init exposed via proc-address so prefill runs at the right time.

---

## [0.7.2] - 2026-08-09

### Added
- **Dynamic pin set (low-memory mode)** — opt-in via `MOE_L2_LRU=1`: register only the activated experts (per-expert group registration) instead of the whole expert tensor; the LRU evictor unregisters + madvises cold experts; layered pin (`MOE_L2_PIN_LAYERS`) keeps universal/sparse layers resident forever (V4 L0-L2 + sparse layers = ~5.4 GB free)
- Measured on RTX 4090 / DeepSeek-V4-Flash (85 GB): RSS **84 GB → 17-24 GB** (`MOE_L2_LRU_MAX_EXPERTS` 2000 ≈ 17 GB / 12000 ≈ 24 GB); speed 4-5.3 t/s (V4 routes extremely wide — new experts pay a ~2.1 ms first-touch page-fault = 0.43 ms page-lock + 1.7 ms read; verified physical limit: WILLNEED prefetch ineffective, group-batching ineffective, staged copy slower)
- bins-v0.3.2 multi-arch binaries (sm_61/75/86/89/120a, CUDA 12.8, no NCCL)

### Changed
- **Default remains whole-pin** (v0.3.1 speed baseline — verified no regression on 2080 Ti: Qwen 16.88 vs 16.94 t/s); dynamic pin set is opt-in low-memory mode (~20% slower on Qwen, ~50% on V4)

### Fixed
- copy_experts pin length now constant (es+padding) so the explicit pin and the set_tensor_async fallback pin agree — previously every copy re-unregistered/re-registered (re-faulting) experts
- pin range merge only on overlap (shared page), not adjacency — adjacency merging made the pinned chain grow without bound (93 s/turn storm)
- unpin simplified (drop the whole chain, no re-register segments) — segment re-registration fragmented the registry (was 378 µs/pin, tens of thousands of ranges)
- removed whole-tensor pin in the mul_mat_id A3 path (would re-pin all 82 GB when A3 is enabled)

---

## [0.7.1] - 2026-08-07

### Added
- **On-demand pin main path** — lazy mmap load (zero VRAM) + first-touch merge-registration of the whole expert tensor via `cudaHostRegister`, GPU reads pinned experts directly via PCIe DMA
- A3 expert cache raised to **2048 slots** (512→2048), near-compute-bound on V4 (GPU util 13% → 86%)

### Changed
- Main path upgraded from host-buffer to on-demand pin (fixes CUDA 11.8 cross-register-range `cudaMemcpyAsync` crash; PROBE_READ pages can no longer be registered)

### Measured (RTX 4090)
- Qwen3.6-A3B: **50.2 t/s** @ 2.9 GB VRAM (+400% vs A3 era, beats pre-lazy 46.5)
- DeepSeek-V2-Lite: **37.9 t/s** @ 2.0 GB VRAM (+200%)
- DeepSeek-V4-Flash (157B, 85 GB file): **10.1 t/s** @ 17.4 GB VRAM (5× faster than 1.7-2.0)
- 2080 Ti (multi-arch): Qwen **24.5 t/s** (2× vs old host-buffer 11.15)

### Fixed
- cli.py: removed REGISTER_HOST + EXPERT_CACHE auto-switch logic (based on wrong premise)

---

## [0.7.0] - 2026-08-05

### Added
- **DeepSeek-V4-Flash (157B MoE) support** — 85 GB 3-shard GGUF runs on 2080 Ti (11 GB) and RTX 3080 (10 GB): VRAM 8.3-9.1 GB, RSS capped by expert-page eviction
- **Expert-page eviction v3.1** (`MOE_L2_LRU_MAX_EXPERTS=N`) — fixed-expert-count LRU, keeps at most N hottest experts resident, evicts only coldest overflow; near-zero slowdown on Qwen (-2% vs v2 -24% / v3 -45%)
- Multi-shard GGUF parsing fix (auto-switch to largest shard when shard 1 is metadata-only)

### Fixed
- proxy non-stream forward timeout 30s → 600s (slow V4 models would ReadTimeout)

### Changed
- GitHub Release **bins-v0.3.0**: v3.1 eviction + multi-arch (sm_61/75/86/89/120a, CUDA 12.8)

---

## [0.6.1] - 2026-08-04

### Fixed
- Nested `bin/` extraction path when auto-downloading binaries (issue #1)
- Missing `libnccl.so.2` dependency — now shipped in cuda-libs/

### Changed
- GitHub Release **bins-v0.2.1**

---

## [0.6.0] - 2026-08-03

### Added
- **Multi-architecture binaries** — one binary for all NVIDIA consumer GPUs: sm_61 (GTX 1080) → sm_120a (RTX 50), built with CUDA 12.8, no per-GPU compilation
- 3-GPU × 2-model benchmark report (2080 Ti / 3080 Ti / 5090)

### Changed
- GitHub Release **bins-v0.2.0**

---

## [0.5.1] - 2026-08-02

### Added
- `moe-l2 doctor` — environment self-check (GPU/CUDA/Python/disk)
- `moe-l2 model list` / `moe-l2 model download` — hf-mirror resumable model downloads
- `scripts/install.sh` — one-line installer (`curl -fsSL ... | bash`): detect → pip install → download-bins → optional model → doctor

---

## [0.5.0] - 2026-08-02

### Added
- **Host-buffer expert GPU fast path** — expert tensors in CUDA host buffer (CPU pinned, zero VRAM), scheduler copies only activated experts to GPU per step; DS 12.5 → 37.5 t/s, Qwen 10 → 46.8 t/s at 1.6 / 2.1 GB VRAM
- sched-cache hooked into scheduler copy layer: DS Prompt +211% (99 → 308 t/s), Gen +5% (cache=0.25)

### Changed
- Binaries moved out of PyPI wheel → GitHub Release **bins-v0.1.1** (fetched via `download-bins`)

---

## [0.4.0] - 2026-08-02

### Added
- `moe-l2 collect` — mode-A routing-data collection (8-domain prompts, generates `domain_expert_map.json` + `meta.json` to `~/.moe-l2/maps/<model_id>/`)
- GGUF metadata embedding (`embed-map` command, custom key `moe_l2.domain_expert_map`)

### Changed
- setuptools >=77 for metadata 2.4 (License-File compatibility)

---

## [0.3.0] - 2026-07-27

### Added
- GPU mode (`moe-l2 start --gpu`) with A3 expert offload patch — DS-V2-Lite VRAM 23.3 → 1.2 GB (-95%), Qwen3.6 7.6 → 2.2 GB (-71%)
- `moe-l2 download-bins` — auto-fetch pre-built CUDA binaries from GitHub Release
- A3 LRU expert cache (C++ layer, 30/30 benchmark PASS)

---

## [0.2.0] - 2026-07-26

### Added
- Domain predictor (hybrid: keyword → TF-IDF → semantic embedding), 8 domains
- L2 cache manager (mmap shared memory, per-layer LRU, async preload)
- GGUF weight reader (zero-copy)
- Transparent proxy (localhost:11435, OpenAI-compatible, SSE streaming)
- CLI: `start` / `stats` / `stop`

---

[0.10.0]: https://github.com/yalun753/moe-l2/releases/tag/bins-v0.6.0
[0.9.0]: https://github.com/yalun753/moe-l2/releases/tag/bins-v0.5.0
[0.8.1]: https://github.com/yalun753/moe-l2/releases/tag/bins-v0.4.1
[0.7.1]: https://github.com/yalun753/moe-l2/releases/tag/bins-v0.3.1
[0.7.0]: https://github.com/yalun753/moe-l2/releases/tag/bins-v0.3.0
[0.6.1]: https://github.com/yalun753/moe-l2/releases/tag/bins-v0.2.1
[0.6.0]: https://github.com/yalun753/moe-l2/releases/tag/bins-v0.2.0
[0.5.1]: https://github.com/yalun753/moe-l2/releases/tag/bins-v0.1.1
[0.5.0]: https://github.com/yalun753/moe-l2/releases/tag/bins-v0.1.1
[0.4.0]: https://pypi.org/project/moe-l2/
[0.3.0]: https://pypi.org/project/moe-l2/
[0.2.0]: https://pypi.org/project/moe-l2/
