#pragma once

#include <cuda_runtime.h>
#include <cstddef>

// LRU Expert Cache for MoE weight offloading.
//
// Caches GPU copies of recently-used expert weight slices so that
// consecutive inference steps hitting the same experts can skip H2D
// transfers. The cache is host-managed (no GPU kernels) and uses a
// simple timestamp-based LRU eviction policy.
//
// Key: cpu_source_address (or (cpu_src, expert_id) pair).
//
// The cache must be initialized before use and freed at context
// destruction (see ggml_cuda_free_data).

// Maximum number of cache slots (safety limit).
// [moe-l2 2026-08-13] raised 512 -> 16384: the slot formula is
// fraction * n_expert * (3 * n_layers); for Qwen (256 experts, 40 layers,
// ~1040 expert accesses/token) the formula yields ~30k slots but the old 512
// cap truncated it, so LRU thrashed every token and the hit rate was 0.0%
// (measured: hit=0 / miss=156189 over 150 tokens). The CLI now injects an
// exact count (MOE_L2_CACHE_SLOTS = n_layers * top_k * 3, e.g. 43*100*3 =
// 12900 for top-k=100), so the cap only guards against runaway configs.
#define EXPERT_CACHE_MAX_SLOTS 32768

// Lazy-init the expert cache from the GGML_CUDA_EXPERT_CACHE env var.
// Slot count is computed as a fraction of the model's per-layer expert
// count (n_expert), so the same setting works for any MoE model:
//   0, "", unset  -> cache disabled (original behaviour)
//   0.25          -> ceil(0.25 * n_expert) slots
//   0.5           -> ceil(0.50 * n_expert) slots
//   0.75          -> ceil(0.75 * n_expert) slots
//   1             -> ceil(1.00 * n_expert) slots (cache all experts)
// n_expert is typically ne02 (number of experts in src0).
// Safe to call multiple times; only the first call with n_expert > 0
// has effect.
void ggml_cuda_expert_cache_maybe_init(int n_expert);

// moe-l2: tell the cache how many transformer layers the model has, so
// slot count can cover the whole model's key space (n_layers x 3 expert
// tensors per layer x n_expert per tensor) instead of one tensor only.
// Safe to call before maybe_init; default 1 (old behaviour) if never called.
void ggml_cuda_expert_cache_set_n_layers(int n_layers);

// Initialize the LRU expert cache for a given device.
//   device           - CUDA device ordinal
//   n_slots          - number of cache slots (clamped to EXPERT_CACHE_MAX_SLOTS)
//   expert_size_bytes- byte size of a single expert weight slice
// Returns 0 on success, -1 on error.
void ggml_cuda_expert_cache_init(int device, int n_slots, size_t expert_size_bytes);

// Look up an expert in the cache by CPU source address.
// If found, returns the cached GPU device pointer.
// If not found, returns NULL.
const void * ggml_cuda_expert_cache_get(const void * cpu_src);

// [moe-l2 sched-cache 2026-08-02] Try to copy an expert from the cache into a
// destination GPU buffer. Used by the scheduler input-copy path (ggml-backend.cpp)
// so hot experts are copied D2D from the cache instead of over PCIe from CPU RAM.
//   cache_key   - stable key (tensor-name hash ^ expert_idx), same as ggml-cuda.cu
//   dst_gpu     - destination GPU device pointer (e.g. input_cpy->data + offset)
//   dst_offset  - byte offset into dst_gpu (usually 0 since dst_gpu already offset)
//   size        - bytes to copy (single expert weight slice)
//   stream      - CUDA stream
// Returns true on hit (data copied from cache), false on miss (caller does the
// regular CPU→GPU copy and then stores via ggml_cuda_expert_cache_set).
bool ggml_cuda_expert_cache_copy_if_hit(
        const void * cache_key,
        void * dst_gpu, size_t dst_offset,
        size_t size, cudaStream_t stream);

// Store an expert weight GPU buffer in the cache.
//   cpu_src    - CPU pointer to the expert weight data (used as key)
//   size       - byte size of the GPU buffer
//   dev_ptr    - GPU device pointer to cache (ownership not transferred)
//   stream     - CUDA stream for any required async operations
// Returns the stored device pointer, or NULL if not cached.
const void * ggml_cuda_expert_cache_set(const void * cpu_src, size_t size, const void * dev_ptr, cudaStream_t stream);

// Free all cached GPU buffers and reset the cache to uninitialized
// state. Safe to call even if the cache was never initialized.
void ggml_cuda_expert_cache_free(void);

// [moe-l2 route-by-domain C 2026-08-15] 运行时调整 cache 槽数：
// 清空释放全部槽（旧领域数据作废），把 n_slots 设为 new_n_slots。
void ggml_cuda_expert_cache_resize(int new_n_slots);

// [moe-l2 retain-hot-experts v1 2026-08-16] 软调整 cache 槽数：
// 只改 n_slots 容量，**不清空已有槽**（换表保留热专家用）。
void ggml_cuda_expert_cache_soft_resize(int new_n_slots);
