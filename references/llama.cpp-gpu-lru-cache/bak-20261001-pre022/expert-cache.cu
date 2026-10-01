#include "expert-cache.cuh"
#include "common.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <new>
#include <thread>

//
// Internal LRU cache structure
//

// [moe-l2 2026-08-19 per-slot lock] ExpertCacheSlot 增加 per-slot 自旋锁。
// 背景：08-13 P0 修复给全局 g_cache 加 std::mutex 后 DS 并发正确（不垃圾），
// 但 Qwen 速度 35.64 -> 30.87 t/s（-10%）。损耗主因不是 mutex 的 futex 系统调用，
// 而是全局锁把 get 的高频线性扫描（每 token 数百次 get × 上万槽）串行化；
// 08-14/15 迭代又把这把锁整个弄丢（生产 bins-v0.5.0 = 无锁版，DS 并发竞态回归）。
// 本版用 per-slot spinlock 替代全局 mutex：
//   - get 无锁预扫描（atomic valid + cpu_src 匹配），命中后才锁单槽（重验 + 更新 LRU）
//   - set 无锁选 LRU victim，锁槽后 double-check timestamp（扫描后被并发动过则重选）
//   - 不同 slot 的 get/set 完全并行；CUDA 调用（free/alloc/copy）只阻塞本槽竞争者
// 正确性模型：get 命中会更新 timestamp（原子 clock 递增），使该槽成为最新
// -> 调用方使用 dev_ptr 期间不会被选为 victim（LRU 语义隐含 lifetime 保护）。
// 生命周期操作（init/free/resize/soft_resize）锁全部槽，与进行中的 get/set 互斥。
struct ExpertCacheSlot {
    const void * cpu_src;   // CPU base address (for key matching)
    char *       dev_ptr;   // cached GPU buffer (owns the allocation)
    size_t       size;      // byte size of this slot's allocation
    uint64_t     timestamp; // LRU timestamp (larger = more recently used)
    std::atomic<bool> valid; // slot is occupied (atomic: get 无锁预扫描与 set 填槽配对)
    std::atomic_flag lock;  // [moe-l2 2026-08-19] per-slot spinlock
};

// ── cache state ────────────────────────────────────────────────
// Singleton: one cache per process.

static struct {
    ExpertCacheSlot slots[EXPERT_CACHE_MAX_SLOTS];
    std::atomic<int>    n_slots;         // active slot count (<= EXPERT_CACHE_MAX_SLOTS)
    int                 device;          // CUDA device ordinal
    std::atomic<bool>   initialized;     // cache has been set up
    std::atomic<uint64_t> clock;         // monotonically increasing timestamp (atomic: concurrent bump)
} g_cache;

// moe-l2: model layer count, set by ggml_cuda_expert_cache_set_n_layers().
// Slot formula uses it to cover the whole-model key space
// (n_layers x 3 expert tensors x n_expert). Default 1 = old behaviour.
static int g_n_layers = 1;

//
// Helpers
//

// [moe-l2 2026-08-19] per-slot spinlock：持锁时间短（元数据读写），
// 用自旋免 futex 系统调用；yield 在低竞争下开销可忽略。
static inline void slot_lock(std::atomic_flag & f) {
    while (f.test_and_set(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

static inline void slot_unlock(std::atomic_flag & f) {
    f.clear(std::memory_order_release);
}

static void expert_cache_clear_slot(ExpertCacheSlot & slot) {
    if (slot.dev_ptr) {
        CUDA_CHECK(cudaFree(slot.dev_ptr));
    }
    slot.cpu_src   = nullptr;
    slot.dev_ptr   = nullptr;
    slot.size      = 0;
    slot.timestamp = 0;
    slot.valid     = false;
}

// [moe-l2 2026-08-19] 生命周期操作入口：锁全部槽（与进行中的 get/set 互斥）。
// 推理中不会调用（resize/soft_resize 在换表请求边界），代价只出现在低频路径。
static void lock_all_slots(int n) {
    for (int i = 0; i < n; i++) {
        slot_lock(g_cache.slots[i].lock);
    }
}

static void unlock_all_slots(int n) {
    for (int i = 0; i < n; i++) {
        slot_unlock(g_cache.slots[i].lock);
    }
}

//
// Public API
//

void ggml_cuda_expert_cache_maybe_init(int n_expert) {
    if (g_cache.initialized.load(std::memory_order_relaxed)) {
        return;
    }
    if (n_expert <= 0) {
        return;
    }

    const char * env = std::getenv("GGML_CUDA_EXPERT_CACHE");
    if (!env || env[0] == '\0') {
        return;
    }

    float fraction = std::atof(env);
    if (fraction <= 0.0f) {
        return;
    }

    // moe-l2: layer count from env var set by llama.cpp at model load
    // (crosses DSO boundary reliably). setter API kept as fallback.
    const char * nl_env = std::getenv("MOE_L2_N_LAYERS");
    if (nl_env && nl_env[0] != '\0') {
        int nl = std::atoi(nl_env);
        if (nl > 0) {
            g_n_layers = nl;
        }
    }

    // moe-l2: allow up to 8x for large caches (512 slots max) so hot experts
    // survive LRU pressure from the ~468 block accesses per token
    if (fraction > 8.0f) {
        fraction = 8.0f;
    }

    // [moe-l2 2026-08-14 cache 容量联动] 首选：路由表元数据 EXPERT_TOTAL。
    // 设计（TOP N / TOP K）：cache 只服务路由表选中的专家，容量 = 选中数 x 3
    // （gate/up/down 每专家 3 张量）。这替代 fraction x 全模型专家空间公式——
    // 后者对 157B 级模型（256 专家 x 41 层）会算出 3 万+ 槽位，clamp 到 2048
    // 后每槽 24MB（Q4）也远超 24GB 显存（实测 OOM）。路由表是 Python 层按
    // 显存预算（N）+ 覆盖率（top-k）收敛好的，用它算槽数 = 回到设计本意。
    // [moe-l2 2026-08-15] 手动槽数优先：MOE_L2_CACHE_SLOTS 显式指定时用
    // 环境变量（08-14 版误删此支持，用户无法手动加大槽数；实测 11913 槽 vs
    // 容量联动 5421 槽 = 命中率 87% vs 65% 级别的差异）。否则走 EXPERT_TOTAL。
    const char * slots_env = std::getenv("MOE_L2_CACHE_SLOTS");
    const char * rf_env = std::getenv("MOE_L2_ROUTER_FILE");
    int n_slots = 0;
    if (slots_env && slots_env[0] != '\0') {
        n_slots = std::atoi(slots_env);
    }
    if (n_slots <= 0) {
        if (rf_env && rf_env[0] != '\0') {
            FILE * rf = std::fopen(rf_env, "r");
            if (rf) {
                char line[1024];
                while (std::fgets(line, sizeof(line), rf)) {
                    if (std::strncmp(line, "# EXPERT_TOTAL ", 15) == 0) {
                        n_slots = std::atoi(line + 15) * 3;  // 3 tensors per expert
                        break;
                    }
                }
                std::fclose(rf);
            }
        }
    }

    // 无路由表（或元数据缺失）→ 回退旧公式：fraction x 全模型专家空间。
    // 保守行为：路由表缺失时保持旧行为，不破坏无表场景（N=0 纯 on-demand）。
    if (n_slots <= 0) {
        const float scale = 3.0f * (float)g_n_layers;
        n_slots = (int)std::ceil(fraction * (float)n_expert * scale);
    }
    if (n_slots < 1) {
        n_slots = 1;
    }
    if (n_slots > EXPERT_CACHE_MAX_SLOTS) {
        n_slots = EXPERT_CACHE_MAX_SLOTS;
    }

    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        return;
    }

    // size=0: no pre-allocation; each slot allocates on first set.
    ggml_cuda_expert_cache_init(device, n_slots, 0);
    fprintf(stderr, "[A3] expert_cache initialized: %d slots on device %d (fraction=%.2f, n_expert=%d, router_total=%d)\n",
            n_slots, device, fraction, n_expert, rf_env && rf_env[0] != '\0' ? n_slots / 3 : -1);
}

void ggml_cuda_expert_cache_set_n_layers(int n_layers) {
    if (n_layers > 0) {
        g_n_layers = n_layers;
    }
}

void ggml_cuda_expert_cache_init(int device, int n_slots, size_t expert_size_bytes) {
    if (g_cache.initialized.load(std::memory_order_relaxed)) {
        ggml_cuda_expert_cache_free();
    }

    int actual_slots = std::min(n_slots, EXPERT_CACHE_MAX_SLOTS);
    if (actual_slots <= 0) {
        actual_slots = 1;
    }

    int prev_device;
    CUDA_CHECK(cudaGetDevice(&prev_device));
    CUDA_CHECK(cudaSetDevice(device));

    for (int i = 0; i < actual_slots; i++) {
        g_cache.slots[i].cpu_src   = nullptr;
        g_cache.slots[i].dev_ptr   = nullptr;
        g_cache.slots[i].size      = 0;
        g_cache.slots[i].timestamp = 0;
        g_cache.slots[i].valid     = false;

        if (expert_size_bytes > 0) {
            void * ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, expert_size_bytes);
            if (err == cudaSuccess) {
                g_cache.slots[i].dev_ptr = (char *)ptr;
                g_cache.slots[i].size    = expert_size_bytes;
            }
        }
    }

    CUDA_CHECK(cudaSetDevice(prev_device));

    g_cache.n_slots.store(actual_slots, std::memory_order_relaxed);
    g_cache.device               = device;
    g_cache.initialized.store(true, std::memory_order_relaxed);
    g_cache.clock.store(0, std::memory_order_relaxed);
}

// [moe-l2 2026-08-19 per-slot lock] 高频路径：无锁预扫描，命中后锁单槽重验。
// 返回的 dev_ptr 生命周期由 LRU 隐含保护（get 已把该槽 timestamp 更新为最新，
// 调用方使用期间不会被 set 选为 victim 淘汰）。
const void * ggml_cuda_expert_cache_get(const void * cpu_src) {
    if (!g_cache.initialized.load(std::memory_order_relaxed) || !cpu_src) {
        return nullptr;
    }

    const int n = g_cache.n_slots.load(std::memory_order_relaxed);

    // Lock-free probe: valid.load(acquire) 与 set 填槽的 valid.store(release) 配对，
    // 保证看到 valid=true 时 cpu_src 的写入完整可见。
    for (int i = 0; i < n; i++) {
        ExpertCacheSlot & slot = g_cache.slots[i];
        if (slot.valid.load(std::memory_order_acquire) && slot.cpu_src == cpu_src) {
            slot_lock(slot.lock);
            // 重验：等待锁期间可能被并发 set 淘汰/改写（LRU victim）
            if (slot.valid.load(std::memory_order_relaxed) && slot.cpu_src == cpu_src) {
                slot.timestamp = g_cache.clock.fetch_add(1, std::memory_order_relaxed) + 1;
                const void * p = slot.dev_ptr;
                slot_unlock(slot.lock);
                return p;
            }
            slot_unlock(slot.lock);
            return nullptr; // 被并发淘汰 → 按 miss 处理（调用方走 H2D 拷贝，正确只是慢）
        }
    }
    return nullptr; // miss
}

bool ggml_cuda_expert_cache_copy_if_hit(
        const void * cache_key,
        void * dst_gpu, size_t dst_offset,
        size_t size, cudaStream_t stream) {
    if (!g_cache.initialized.load(std::memory_order_relaxed) || !cache_key || !dst_gpu || size == 0) {
        return false;
    }

    const void * src = ggml_cuda_expert_cache_get(cache_key);
    if (src == nullptr) {
        return false; // miss — caller falls back to CPU→GPU copy
    }

    // Hit: D2D copy from the cache slot into the destination GPU buffer.
    // The cache slot's device pointer is owned by the cache and stays valid
    // across requests (unlike sched input-copy buffers).
    {
        // [moe-l2 debug] log pointer types once to diagnose invalid-argument
        static int dbg = 0;
        if (dbg < 6) {
            cudaPointerAttributes pa_dst, pa_src;
            cudaError_t e_dst = cudaPointerGetAttributes(&pa_dst, (char *)dst_gpu + dst_offset);
            cudaError_t e_src = cudaPointerGetAttributes(&pa_src, src);
            fprintf(stderr, "[SCACHE-DBG] dst=%p dst_attr_err=%d type=%d src=%p src_err=%d type=%d size=%zu\n",
                (char *)dst_gpu + dst_offset, (int)e_dst, e_dst == cudaSuccess ? (int)pa_dst.type : -1,
                src, (int)e_src, e_src == cudaSuccess ? (int)pa_src.type : -1, size);
            dbg++;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(
        (char *)dst_gpu + dst_offset, src, size,
        cudaMemcpyDeviceToDevice, stream));
    return true;
}

// [moe-l2 2026-08-19 per-slot lock] 低频路径（仅 miss 时）：
// 无锁选 LRU victim → 锁槽 double-check（扫描后被并发 get/set 动过则重选，最多 16 次）
// → evict/alloc/async copy/填槽 → 解锁。
// double-check 保证：victim 在扫描后未被任何线程碰过（timestamp 未变），
// 因此该槽没有正在使用中的调用方 → cudaFree 旧指针安全。
const void * ggml_cuda_expert_cache_set(const void * cpu_src, size_t size, const void * dev_ptr, cudaStream_t stream) {
    if (!g_cache.initialized.load(std::memory_order_relaxed) || !cpu_src || !dev_ptr || size == 0) {
        return nullptr;
    }

    const int n = g_cache.n_slots.load(std::memory_order_relaxed);

    const int MAX_ATTEMPTS = 16;
    for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        // ── Find LRU victim（无锁读）──
        int victim = -1;
        uint64_t victim_ts = 0;
        bool victim_valid = false;
        uint64_t oldest = UINT64_MAX;

        for (int i = 0; i < n; i++) {
            ExpertCacheSlot & slot = g_cache.slots[i];
            const bool v = slot.valid.load(std::memory_order_relaxed);
            const uint64_t ts = slot.timestamp;
            if (!v) {
                victim = i;
                victim_ts = ts;
                victim_valid = false;
                break;
            }
            if (ts < oldest) {
                oldest = ts;
                victim = i;
                victim_ts = ts;
                victim_valid = true;
            }
        }

        if (victim < 0) {
            return nullptr;
        }

        ExpertCacheSlot & slot = g_cache.slots[victim];
        slot_lock(slot.lock);

        // ── Double-check：扫描后是否被并发动过 ──
        const bool v_now = slot.valid.load(std::memory_order_relaxed);
        const uint64_t ts_now = slot.timestamp;
        if (v_now != victim_valid || ts_now != victim_ts) {
            slot_unlock(slot.lock);
            continue; // 被并发 get/set 碰过 → 重选 victim
        }

        // ── Evict old entry ──
        if (v_now) {
            slot.cpu_src = nullptr;
            slot.valid.store(false, std::memory_order_relaxed); // 先 invalid，防并发 get 命中旧 cpu_src
        }

        // ── Ensure device buffer is large enough ──
        if (slot.size < size) {
            if (slot.dev_ptr) {
                CUDA_CHECK(cudaFree(slot.dev_ptr));
            }
            int prev_device;
            CUDA_CHECK(cudaGetDevice(&prev_device));
            CUDA_CHECK(cudaSetDevice(g_cache.device));

            void * ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, size);
            CUDA_CHECK(cudaSetDevice(prev_device));

            if (err != cudaSuccess) {
                slot.dev_ptr = nullptr;
                slot.size    = 0;
                slot_unlock(slot.lock);
                return nullptr;
            }
            slot.dev_ptr = (char *)ptr;
            slot.size    = size;
        }

        // ── Async D2D copy ──
        CUDA_CHECK(cudaMemcpyAsync(
            slot.dev_ptr, dev_ptr, size,
            cudaMemcpyDeviceToDevice, stream));

        // ── Fill slot metadata（写序：先 cpu_src，后 valid.release 发布）──
        slot.cpu_src   = cpu_src;
        slot.timestamp = g_cache.clock.fetch_add(1, std::memory_order_relaxed) + 1;
        slot.valid.store(true, std::memory_order_release);

        const void * stored = slot.dev_ptr; // 锁内取值，解锁后返回
        slot_unlock(slot.lock);
        return stored;
    }

    // 高竞争下 16 次仍失败 → 放弃缓存本次（调用方 fallback 用临时 GPU 缓冲，正确）
    return nullptr;
}

void ggml_cuda_expert_cache_free(void) {
    if (!g_cache.initialized.load(std::memory_order_relaxed)) {
        return;
    }

    const int n = g_cache.n_slots.load(std::memory_order_relaxed);
    lock_all_slots(n);

    int prev_device;
    CUDA_CHECK(cudaGetDevice(&prev_device));
    CUDA_CHECK(cudaSetDevice(g_cache.device));

    for (int i = 0; i < n; i++) {
        expert_cache_clear_slot(g_cache.slots[i]);
    }

    CUDA_CHECK(cudaSetDevice(prev_device));

    // 注意：slots 含 atomic 字段（lock/valid），不能用 memset 清零（UB）。
    // clear_slot 已重置非 atomic 字段；atomic 字段用 store/clear 显式复位。
    for (int i = 0; i < n; i++) {
        g_cache.slots[i].valid.store(false, std::memory_order_relaxed);
        g_cache.slots[i].lock.clear(std::memory_order_release);
    }
    g_cache.initialized.store(false, std::memory_order_relaxed);
    g_cache.n_slots.store(0, std::memory_order_relaxed);
    g_cache.clock.store(0, std::memory_order_relaxed);

    unlock_all_slots(n);
}

// [moe-l2 route-by-domain C 2026-08-15] 槽数随路由表动态调整。
// 换表（/moe-set-domain）时调用：清空并释放全部槽（旧领域数据作废），
// 把 n_slots 设为 new_n_slots（clamp 到 [1, EXPERT_CACHE_MAX_SLOTS]）。
// 解决：flywheel 表 rebuild 变大后换表 payload 超过启动时定的槽数，
// LRU 反复逐出且不释放显存 → DS 大专家场景显存峰值超限 OOM。
void ggml_cuda_expert_cache_resize(int new_n_slots) {
    if (!g_cache.initialized.load(std::memory_order_relaxed)) {
        return;
    }

    const int n = g_cache.n_slots.load(std::memory_order_relaxed);
    lock_all_slots(n);

    int prev_device;
    CUDA_CHECK(cudaGetDevice(&prev_device));
    CUDA_CHECK(cudaSetDevice(g_cache.device));

    for (int i = 0; i < n; i++) {
        expert_cache_clear_slot(g_cache.slots[i]);
    }

    CUDA_CHECK(cudaSetDevice(prev_device));

    int actual = std::min(new_n_slots, EXPERT_CACHE_MAX_SLOTS);
    if (actual < 1) {
        actual = 1;
    }
    g_cache.n_slots.store(actual, std::memory_order_relaxed);
    g_cache.clock.store(0, std::memory_order_relaxed);
    fprintf(stderr, "[A3] expert_cache resized: %d slots on device %d\n",
            g_cache.n_slots.load(std::memory_order_relaxed), g_cache.device);

    unlock_all_slots(n);
}

// [moe-l2 retain-hot-experts v1 2026-08-16] 软调整 cache 槽数：
// 只改 n_slots 容量，**不清空已有槽**（区别于 resize 的全清）。
// 用途：换表保留热专家——换新领域表时旧领域专家继续留在 cache，
// 新表 prefill 时 get 命中已存在的专家就跳过，避免重复拷贝和互相挤占。
// 扩大直接扩（后续 set 会填充）；缩小只降上限，超出的槽由 LRU 访问自然逐出。
void ggml_cuda_expert_cache_soft_resize(int new_n_slots) {
    if (!g_cache.initialized.load(std::memory_order_relaxed)) {
        return;
    }

    const int n = g_cache.n_slots.load(std::memory_order_relaxed);
    lock_all_slots(n);

    int actual = std::min(new_n_slots, EXPERT_CACHE_MAX_SLOTS);
    if (actual < 1) {
        actual = 1;
    }
    g_cache.n_slots.store(actual, std::memory_order_relaxed);
    fprintf(stderr, "[A3] expert_cache soft_resized: %d slots on device %d (keep existing)\n",
            g_cache.n_slots.load(std::memory_order_relaxed), g_cache.device);

    unlock_all_slots(n);
}
