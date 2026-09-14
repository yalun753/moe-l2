"""L2 hot-cache manager for moe-l2.

Manages a shared-memory pool of preloaded expert weights with:
  - mmap-backed shared memory in /dev/shm/ for zero-copy access
  - Per-layer LRU eviction with pinned expert support
  - Asynchronous preloading from domain predictions
  - Thread-safe operations (protected by RLock)
  - Hit/miss statistics tracking

Architecture:
  L2 (RAM) sits between L3 (SSD/GGUF file) and L0 (GPU pool).
  Experts that are likely to be needed are preloaded into L2
  based on domain predictions, so they can be memcpy'd into
  GPU pool in ~1150 µs instead of ~6500 µs from SSD.

Usage:
    cache = L2Cache()
    cache.pin_experts(0, [41, 72, 89])  # backbone experts

    # Preload after prediction
    cache.preload_domain("codegen", expert_map)

    # On each token's expert selection
    hot = cache.request(layer=15, expert_id=42)

    # Check stats
    print(cache.stats())
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from .gguf_reader import MoEGGUFReader

# ── Defaults ──
DEFAULT_N_LAYERS = 40
DEFAULT_SLOTS_PER_LAYER = 96
DEFAULT_EXPERT_SIZE = 1_010_000  # ~1.01 MB (fallback when no model_path)
DEFAULT_L2_DIR = Path("/dev/shm/moe_l2")
DEFAULT_LOADER_WORKERS = 2

# Sentinels for slot tracking
_EMPTY = -1        # slot is free
_RESERVED = -2     # slot is being loaded (between evict and write)


class L2Cache:
    """Per-layer LRU cache of expert weights in POSIX shared memory.

    Each slot is an mmap'd file under {l2_dir}/. Filename: L{layer}_S{slot}.
    Content: raw float16 expert weights (expert_size bytes).

    Design choices (from LRU simulation + 8-domain expert analysis):
      - 96 slots/layer: 84.4% hit rate across 8 domains (near ceiling)
      - Per-layer independent LRU: layer variances were only ~7pp
      - Pinned backbone/domain experts: small benefit at 96-slots,
        critical at smaller capacities
      - Async loading: non-blocking L3→L2 during user thinking time

    Thread safety: all public methods use RLock internally.
    """

    def __init__(
        self,
        n_layers: Optional[int] = None,
        slots_per_layer: int = DEFAULT_SLOTS_PER_LAYER,
        expert_size: Optional[int] = None,
        l2_dir: Optional[Path] = None,
        loader_workers: int = DEFAULT_LOADER_WORKERS,
        expert_data_dir: Optional[Path] = None,
        model_path: Optional[str | Path] = None,
    ):
        # ── Auto-detect from GGUF model ──
        self._gguf_reader: Optional[MoEGGUFReader] = None
        if model_path is not None:
            self._gguf_reader = MoEGGUFReader(model_path)
            self.n_layers = n_layers or self._gguf_reader.num_layers
            self.expert_size = (
                expert_size or self._gguf_reader.per_expert_size()
            )
            self._num_experts = self._gguf_reader.num_experts
        else:
            self.n_layers = n_layers or DEFAULT_N_LAYERS
            self.expert_size = expert_size or DEFAULT_EXPERT_SIZE
            self._num_experts = None

        self.slots_per_layer = slots_per_layer
        self.l2_dir = l2_dir or DEFAULT_L2_DIR
        self.expert_data_dir = expert_data_dir

        # Create shared memory directory
        self.l2_dir.mkdir(parents=True, exist_ok=True)

        # ── Slot state ──
        # _slots[layer][slot] -> expert_id or _EMPTY
        # _lru_order[layer] -> list[expert_id]; most recent at end
        # _pinned[layer] -> set[expert_id]; never evictable
        # _domain_pinned[layer] -> set[expert_id]; pinned by active domain
        self._slots: list[list[int]] = [
            [_EMPTY] * slots_per_layer for _ in range(self.n_layers)
        ]
        self._lru_order: list[list[int]] = [[] for _ in range(self.n_layers)]
        self._pinned: list[set[int]] = [set() for _ in range(self.n_layers)]
        self._domain_pinned: list[set[int]] = [set() for _ in range(self.n_layers)]

        # ── Statistics ──
        self._hits: int = 0
        self._misses: int = 0
        self._loads_completed: int = 0

        # ── Threading ──
        self._lock = threading.RLock()
        self._loader = ThreadPoolExecutor(
            max_workers=loader_workers,
            thread_name_prefix="moe-l2-loader",
        )
        # Track in-flight loads: (layer, expert_id) -> Future
        self._pending_loads: dict[tuple[int, int], Future] = {}

        # ── Active domain tracking ──
        self._active_domain: Optional[str] = None

    # ═══════════════════════════════════════════════════════════════
    # Core API
    # ═══════════════════════════════════════════════════════════════

    def request(self, layer: int, expert_id: int) -> bool:
        """Request an expert. Returns True if cached (hot), False if miss.

        On hit: marks the expert as MRU (moved to LRU tail).
        On miss: submits async L3→L2 load if not already pending.
        """
        with self._lock:
            if self._is_cached(layer, expert_id):
                self._touch(layer, expert_id)
                self._hits += 1
                return True

            self._misses += 1
            if (layer, expert_id) not in self._pending_loads:
                future = self._loader.submit(
                    self._load_expert, layer, expert_id
                )
                self._pending_loads[(layer, expert_id)] = future
            return False

    def request_batch(self, requests: list[tuple[int, int]]) -> dict:
        """Request multiple experts at once.

        Args:
            requests: list of (layer, expert_id) pairs.

        Returns:
            dict with keys: hits, misses, hit_count, miss_count
        """
        hits: list = []
        misses: list = []
        with self._lock:
            for layer, expert_id in requests:
                if self._is_cached(layer, expert_id):
                    self._touch(layer, expert_id)
                    self._hits += 1
                    hits.append((layer, expert_id))
                else:
                    self._misses += 1
                    if (layer, expert_id) not in self._pending_loads:
                        future = self._loader.submit(
                            self._load_expert, layer, expert_id
                        )
                        self._pending_loads[(layer, expert_id)] = future
                    misses.append((layer, expert_id))

        return {
            "hits": hits,
            "misses": misses,
            "hit_count": len(hits),
            "miss_count": len(misses),
        }

    def preload_domain(
        self, domain: str, expert_map: dict
    ) -> None:
        """Preload all experts for a predicted domain (smooth domain switch).

        平滑过渡策略（2026-08-02 定稿：LRU 之上的策略层，非重写缓存）：
        - 不主动 clear 旧域 pin：旧域专家靠 LRU 自然衰减（最久未用先走），
          不做"切换领域→清空"（避免冷启动全部 miss）
        - 新域专家按需 pin + 加载
        - 骨干专家（_pinned）天然保留，不参与淘汰

        Args:
            domain: Domain label (e.g. "codegen", "math").
            expert_map: Domain→expert mapping dict (from predictor._get_map()).
        """
        domain_data = expert_map.get("domains", {}).get(domain, {})
        per_layer_top = dict(domain_data.get("per_layer_top", {}))
        if not per_layer_top:
            return
        # 全覆盖兜底（2026-09-14）：C++ EXPERT_LOG 可能未覆盖后段层
        # （如 48 层模型只记前 40 层）→ 路由表缺失这些层 → f4 空白。
        # 对缓存探测到的真实层、路由表未含的层，盲补该层 top-k 专家，
        # 让 L2Cache 接管全部 MoE 层（无 model_path 不生效）。
        self._backfill_missing_layers(per_layer_top)

        with self._lock:
            # 平滑切换：旧域专家降级为 LRU 可淘汰（仍留在缓存，靠 LRU 自然衰减），
            # 新域专家 pin。不清空缓存，避免冷启动全部 miss；
            # 不保留旧域 pin，避免 pin 累积占满 slots 使 LRU 失效。
            self._active_domain = domain

            # 旧域 pin 全部降级（保留在 slots，但不再占 pin 名额）。
            # 注意：必须先 clear 再 _touch —— _touch 对仍在 pin 集合里的
            # 专家会直接 return，若在 clear 之前调用会导致旧域专家既失去
            # pin 又不进入 LRU 追踪（幽灵槽位），新域专家加载时找不到
            # 可逐出槽位而被静默跳过，缓存切换后即卡死。
            for layer_idx in range(self.n_layers):
                old_pins = list(self._domain_pinned[layer_idx])
                self._domain_pinned[layer_idx].clear()
                for eid in old_pins:
                    if eid not in self._pinned[layer_idx]:
                        self._touch(layer_idx, eid)

            # Collect loads and pin
            to_load: list[tuple[int, int]] = []
            for layer_str, expert_ids in per_layer_top.items():
                layer = int(layer_str)
                if not (0 <= layer < self.n_layers):
                    continue
                for eid in expert_ids:
                    if not self._is_cached(layer, eid):
                        to_load.append((layer, eid))
                    # All domain-preferred experts get domain-pinned
                    self._domain_pinned[layer].add(eid)

            # Submit async loads
            for layer, eid in to_load:
                if (layer, eid) not in self._pending_loads:
                    f = self._loader.submit(self._load_expert, layer, eid)
                    self._pending_loads[(layer, eid)] = f

    def _backfill_missing_layers(self, per_layer_top: dict) -> None:
        """覆盖率兜底：对缓存探测到的真实层、但路由表缺失的层，盲补 top-k 专家。

        不依赖领域路由统计，按各层最大专家数（与其他层一致）盲选 0..k-1。
        无 model_path（_num_experts=None）或层已全部覆盖时为空操作。
        """
        if self._num_experts is None or self._num_experts <= 0:
            return
        k = max((len(v) for v in per_layer_top.values()), default=self.slots_per_layer)
        k = min(k, self.slots_per_layer, self._num_experts)
        for layer in range(self.n_layers):
            if layer not in per_layer_top:
                per_layer_top[layer] = list(range(k))

    def promote_domain(self, domain: str, expert_map: dict) -> None:
        """预热：检测到路由漂移时抬高目标域专家的 LRU 优先级。

        平滑过渡策略的"预热"动作——不重写缓存，只是把目标域
        已缓存的专家手动移到 LRU 尾部（MRU），让它们在淘汰时
        排最后；未缓存的走正常异步加载。

        Args:
            domain: Target domain label.
            expert_map: Domain→expert mapping dict.
        """
        domain_data = expert_map.get("domains", {}).get(domain, {})
        per_layer_top = domain_data.get("per_layer_top", {})
        if not per_layer_top:
            return

        with self._lock:
            self._active_domain = domain
            to_load: list[tuple[int, int]] = []
            for layer_str, expert_ids in per_layer_top.items():
                layer = int(layer_str)
                if not (0 <= layer < self.n_layers):
                    continue
                for eid in expert_ids:
                    if self._is_cached(layer, eid):
                        # 已缓存 → 抬高 LRU 优先级（MRU），不 pin 也可保留
                        self._touch(layer, eid)
                    else:
                        to_load.append((layer, eid))
            # 未缓存的正常异步加载
            for layer, eid in to_load:
                if (layer, eid) not in self._pending_loads:
                    f = self._loader.submit(self._load_expert, layer, eid)
                    self._pending_loads[(layer, eid)] = f

    def preload_batch(self, pairs: list[tuple[int, int]]) -> None:
        """Preload a specific list of (layer, expert_id) pairs."""
        with self._lock:
            for layer, expert_id in pairs:
                if not self._is_cached(layer, expert_id) \
                        and (layer, expert_id) not in self._pending_loads:
                    f = self._loader.submit(self._load_expert, layer, expert_id)
                    self._pending_loads[(layer, expert_id)] = f

    def contains(self, layer: int, expert_id: int) -> bool:
        """Check if an expert is already in L2 cache."""
        with self._lock:
            return self._is_cached(layer, expert_id)

    def pin_experts(self, layer: int, expert_ids: list[int]) -> None:
        """Permanently pin backbone/universal experts (never evicted).

        These are experts activated across ALL domains — they should
        never be evicted from cache.
        """
        with self._lock:
            self._pinned[layer].update(expert_ids)

    def unpin_experts(self, layer: int, expert_ids: list[int]) -> None:
        """Remove permanent pin status from experts."""
        with self._lock:
            self._pinned[layer].difference_update(expert_ids)

    def clear_domain_pins(self) -> None:
        """Remove all domain-level pins (called on domain switch)."""
        with self._lock:
            for layer in range(self.n_layers):
                self._domain_pinned[layer].clear()
            self._active_domain = None

    def wait_for_pending(self, timeout: float = 30.0) -> int:
        """Wait for all pending expert loads to complete.

        Args:
            timeout: Max seconds to wait.

        Returns:
            Number of completed loads.
        """
        with self._lock:
            pending = list(self._pending_loads.values())

        done = 0
        for f in pending:
            try:
                f.result(timeout=timeout)
                done += 1
            except Exception:
                pass
        return done

    def get_slot_path(self, layer: int, slot: int) -> Optional[Path]:
        """Get the shared memory file path for a slot.

        Returns None if invalid layer/slot indices.
        """
        with self._lock:
            if 0 <= layer < self.n_layers and 0 <= slot < self.slots_per_layer:
                return self.l2_dir / f"L{layer}_S{slot}"
        return None

    def get_expert_slot(self, layer: int, expert_id: int) -> Optional[int]:
        """Get the slot index for a cached expert, or None if not cached."""
        with self._lock:
            for slot, eid in enumerate(self._slots[layer]):
                if eid == expert_id:
                    return slot
        return None

    def stats(self) -> dict:
        """Return cache statistics.

        Includes: hits, misses, hit rate, utilization, memory usage,
        pending loads, active domain, and per-layer breakdown.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = round(self._hits / total * 100, 1) if total > 0 else 0.0

            total_slots = self.n_layers * self.slots_per_layer
            used_slots = sum(
                1 for ls in self._slots for s in ls if s != _EMPTY
            )
            pinned_count = sum(
                len(self._pinned[layer] | self._domain_pinned[layer])
                for layer in range(self.n_layers)
            )

            # Per-layer breakdown (compact)
            layer_stats = []
            for layer in range(self.n_layers):
                used = sum(1 for s in self._slots[layer] if s != _EMPTY)
                pinned = len(
                    self._pinned[layer] | self._domain_pinned[layer]
                )
                layer_stats.append(
                    f"  L{layer:2d}: {used:3d}/{self.slots_per_layer:3d} slots "
                    f"({pinned:3d} pinned)"
                )

            return {
                "hits": self._hits,
                "misses": self._misses,
                "total_requests": total,
                "hit_rate_pct": hit_rate,
                "total_slots": total_slots,
                "used_slots": used_slots,
                "pinned_experts": pinned_count,
                "pending_loads": len(self._pending_loads),
                "loads_completed": self._loads_completed,
                "utilization_pct": (
                    round(used_slots / total_slots * 100, 1)
                    if total_slots else 0.0
                ),
                "memory_usage_mb": round(
                    used_slots * self.expert_size / (1024 * 1024), 1
                ),
                "active_domain": self._active_domain,
                "per_layer_lines": layer_stats,
            }

    def close(self) -> None:
        """Clean up all shared memory regions and thread pool."""
        self._loader.shutdown(wait=False)

        # Clean up shm files
        for layer in range(self.n_layers):
            for slot in range(self.slots_per_layer):
                path = self._get_shm_path(layer, slot)
                if path.exists():
                    path.unlink()

        # Remove directory if empty
        try:
            self.l2_dir.rmdir()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ═══════════════════════════════════════════════════════════════
    # Internal helpers
    # ═══════════════════════════════════════════════════════════════

    def _get_shm_path(self, layer: int, slot: int) -> Path:
        """Absolute path to a slot's shared memory file."""
        return self.l2_dir / f"L{layer}_S{slot}"

    def _is_cached(self, layer: int, expert_id: int) -> bool:
        """Check if expert weights are in shared memory.

        NOTE: caller MUST hold self._lock.
        """
        if not (0 <= layer < self.n_layers):
            return False
        return expert_id in self._slots[layer]

    def _touch(self, layer: int, expert_id: int) -> None:
        """Mark expert as MRU — move to end of LRU list.

        NOTE: caller MUST hold self._lock.
        Pinned experts are not moved in LRU order.
        """
        if expert_id in self._pinned[layer] or expert_id in self._domain_pinned[layer]:
            return  # pinned: don't track in LRU
        lru = self._lru_order[layer]
        if expert_id in lru:
            lru.remove(expert_id)
        lru.append(expert_id)  # most recent = end

    def _find_evictable_slot(self, layer: int) -> Optional[int]:
        """Find a slot to use for loading.

        Priority: empty slot > LRU unpinned slot.
        Skips _RESERVED slots (in-flighting loads).
        Returns slot index, or None if all slots pinned/reserved.

        NOTE: caller MUST hold self._lock.
        """
        slots = self._slots[layer]
        pinned = self._pinned[layer] | self._domain_pinned[layer]
        lru = self._lru_order[layer]

        # 1. Prefer an already-empty slot (not in-flight)
        for i, eid in enumerate(slots):
            if eid == _EMPTY:
                return i

        # 2. Evict LRU unpinned expert (front of list = least recently used)
        for eid in lru:
            if eid not in pinned:
                for i, seid in enumerate(slots):
                    if seid == eid:
                        return i

        return None  # all pinned/reserved — can't load anything new

    def _evict_slot(self, layer: int, slot: int) -> None:
        """Reserve a slot for loading. Caller MUST hold self._lock.

        Sets the slot to _RESERVED to prevent concurrent loaders
        from taking it. The actual expert_id is set after I/O completes.
        """
        old_eid = self._slots[layer][slot]
        if old_eid not in (_EMPTY, _RESERVED):
            # Remove from LRU tracking
            lru = self._lru_order[layer]
            if old_eid in lru:
                lru.remove(old_eid)
            # Remove the shm file
            path = self._get_shm_path(layer, slot)
            if path.exists():
                path.unlink()
        self._slots[layer][slot] = _RESERVED

    def _load_expert(self, layer: int, expert_id: int) -> None:
        """Load expert weights from disk into a shared memory slot.

        Runs in background thread pool. Sequence:
        1. Reserve a slot (find or evict), set to _RESERVED
        2. Read weights from disk (I/O, outside lock)
        3. Write to shared memory file
        4. Update slot from _RESERVED to expert_id (inside lock)

        On failure: revert slot to _EMPTY.
        """
        slot: Optional[int] = None
        try:
            # Step 1: reserve a slot
            with self._lock:
                # Double-check: loaded while we were queued
                if self._is_cached(layer, expert_id):
                    self._pending_loads.pop((layer, expert_id), None)
                    return

                slot = self._find_evictable_slot(layer)
                if slot is None:
                    # All slots pinned/reserved — silently skip
                    self._pending_loads.pop((layer, expert_id), None)
                    return

                # Evict occupant → set slot to _RESERVED
                self._evict_slot(layer, slot)

            # Step 2-3: I/O outside lock
            data = self._read_expert_weights(layer, expert_id)
            shm_path = self._get_shm_path(layer, slot)
            with open(shm_path, "wb") as f:
                f.write(data)

            # Step 4: finalize (inside lock)
            with self._lock:
                # Only overwrite if still _RESERVED (not stolen)
                if self._slots[layer][slot] == _RESERVED:
                    self._slots[layer][slot] = expert_id
                    self._touch(layer, expert_id)
                self._loads_completed += 1
                self._pending_loads.pop((layer, expert_id), None)

        except Exception:
            # On failure: free the slot
            with self._lock:
                self._pending_loads.pop((layer, expert_id), None)
                if slot is not None \
                        and 0 <= layer < self.n_layers \
                        and 0 <= slot < self.slots_per_layer \
                        and self._slots[layer][slot] == _RESERVED:
                    self._slots[layer][slot] = _EMPTY

    def _read_expert_weights(self, layer: int, expert_id: int) -> bytes:
        """Read expert weights from disk.

        Priority:
          1. GGUF model (via MoEGGUFReader) — if model_path was provided
          2. Pre-extracted .bin file: {expert_data_dir}/L{layer}_E{expert_id}.bin
          3. Zero-filled placeholder (fallback)

        Returns exactly expert_size bytes.
        """
        # Priority 1: read from GGUF model
        if self._gguf_reader is not None:
            data = self._gguf_reader.read_expert_weights(layer, expert_id)
            if len(data) >= self.expert_size:
                return data[:self.expert_size]
            return data + b"\x00" * (self.expert_size - len(data))

        # Priority 2: pre-extracted .bin files
        if self.expert_data_dir:
            bin_path = self.expert_data_dir / f"L{layer}_E{expert_id}.bin"
            if bin_path.exists():
                with open(bin_path, "rb") as f:
                    data = f.read()
                if len(data) >= self.expert_size:
                    return data[:self.expert_size]
                # Pad if shorter
                return data + b"\x00" * (self.expert_size - len(data))

        # Priority 3: placeholder zeros
        return b"\x00" * self.expert_size

    def _pin_backbone_from_map(self, expert_map: dict) -> None:
        """Pin backbone/universal experts from the domain→expert mapping.

        Backbone experts are those activated in ALL tested domains.
        In Qwen3.6 data, the 10 backbone experts are:
          [41, 72, 89, 95, 112, 127, 191, 217, 221, 231]
        (These appear in the top-15 of all 8 domains.)

        This is called once at init time if auto_pin_backbone is set.
        """
        backbone = expert_map.get("backbone_experts", [])
        if backbone:
            for layer in range(self.n_layers):
                self.pin_experts(layer, backbone)
