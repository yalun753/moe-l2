"""路由表数据飞轮（2026-08-09 新增，不改动现有源码）。

原理：gate 已经在实时解析 EXPERT 路由日志（每 token 每层的 expert id）。
把这些真实路由按领域聚合，攒够阈值后自动重建领域路由表
（domain_router_map_*.json 同格式：domains.<domain>.per_layer_domain_preferred），
让 pretouch / 批量 pin 消费的路由表"越用越准"，而不是生成一次就固定。

设计：
- 新增独立模块，不修改 gate.py / proxy.py / cache.py 的既有逻辑
- 由调用方在已有 hook 点（gate.on_log_line / proxy 请求级）追加一行调用
- 聚合计数放内存（Counter），攒够 N 条 EXPERT 记录自动写盘（原子替换）
- 路由表格式与 load_mapping() 消费的 domain_expert_map.json 完全一致，
  所以 pretouch / cache.preload_domain / 批量 pin 无需改动即可消费新表
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger("moe-l2-router-flywheel")

# 默认输出：moe_l2/data/domain_router_map_flywheel_{model_id}.json
# [2026-08-13 flywheel B] flywheel 表按模型分文件：每个模型一张动态表，
# 各模型热度/收敛数据互不干扰（医院=模型，各医院独立收敛）。
# 旧单文件 domain_router_map_flywheel.json 退役（不再读写，留作 legacy）。
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "moe_l2" / "data"
DEFAULT_MAP_PATH = DATA_DIR / "domain_router_map_flywheel.json"  # legacy fallback（无 model_id 时）

# 每层每领域保留的高频专家数（与验证报告 top-75 对齐；Qwen 用 top-100 也兼容）
# [2026-08-16 实验] top_k 75→100：验证提高路由表覆盖能否提升命中率（08-13 大表 3971 专家命中 87.5% vs 3000 专家 71.8%）
DEFAULT_TOP_K = 100
# 触发重建路由表的 EXPERT 记录阈值
REBUILD_EVERY_N = 5000
# [2026-09-19 节流] 重建最小间隔（秒）。5000 条门槛 ≈ 16 个 token 的生成量
# （每 token 40 层 × 8 专家 = 320 条），不节流就会几秒重写一次表 → 每次重建都
# 触发换表/重新预热，表现为“flywheel rebuilt 反复触发”+ 速度抖动（历史待查项）。
# 默认 60s：一分钟最多重建一次；MOE_L2_REBUILD_MIN_INTERVAL 可覆盖。
REBUILD_MIN_INTERVAL_S = 60.0
# 至少需要多少条才开始重建（防止刚启动就用 3 条记录覆盖真实表）
MIN_RECORDS_FOR_REBUILD = 500


class DomainRouterFlywheel:
    """聚合真实门路由记录 → 按领域统计高频专家 → 原子更新路由表 JSON。

    线程安全：proxy 的 GateReaderThread 和请求线程都会调用，用锁保护。
    """

    def __init__(
        self,
        map_path: Path | None = None,
        model_id: str = "",
        top_k: int = DEFAULT_TOP_K,
        rebuild_every: int = REBUILD_EVERY_N,
        min_records: int = MIN_RECORDS_FOR_REBUILD,
        rebuild_min_interval: float | None = None,
    ) -> None:
        # [2026-08-13 flywheel B] 按模型分文件：map_path 未显式指定时，
        # 用 domain_router_map_flywheel_{model_id}.json；无 model_id 时回退 legacy 单文件。
        if map_path is None:
            if model_id:
                map_path = DATA_DIR / f"domain_router_map_flywheel_{model_id}.json"
            else:
                map_path = DEFAULT_MAP_PATH
        self.map_path = Path(map_path)
        self.top_k = top_k
        self.rebuild_every = rebuild_every
        self.min_records = min_records
        if rebuild_min_interval is None:
            try:
                rebuild_min_interval = float(
                    os.environ.get("MOE_L2_REBUILD_MIN_INTERVAL",
                                   REBUILD_MIN_INTERVAL_S))
            except (TypeError, ValueError):
                rebuild_min_interval = REBUILD_MIN_INTERVAL_S
        self.rebuild_min_interval = max(0.0, float(rebuild_min_interval))
        self._last_rebuild = 0.0

        # 当前领域（由请求级信号设置，默认 general）
        self._domain = "general"
        # 聚合计数：domain -> layer -> Counter(expert_id -> 次数)
        self._agg: dict[str, dict[int, Counter]] = defaultdict(
            lambda: defaultdict(Counter))
        # 领域热度：[2026-08-13] domain -> 请求次数（热门领域依据）
        self._dom_freq: Counter = Counter()
        self._records = 0
        self._rebuilds = 0
        self._lock = __import__("threading").Lock()

        # [2026-08-16 load-existing] 启动时加载已有表到 _agg（保底计数），
        # 防止进程重启后新进程的少量请求 rebuild 覆盖丢失历史领域/专家。
        self._load_existing_map()

    def _load_existing_map(self) -> None:
        """把已有路由表 JSON 的 per_layer 专家加载进 _agg（计数=1 保底）。

        背景：_agg 是纯内存聚合，进程重启归零；若直接跑少量请求就 rebuild，
        os.replace 会用新进程数据覆盖全量历史表（实测丢 translate/creative_write
        /math/logic 4 个领域，Qwen 掉速 55%）。加载旧表后 rebuild 时旧领域/专家
        至少保留 top-k 条，新数据自然累积覆盖高频位置，不阻塞"越用越准"。
        """
        try:
            if not self.map_path.exists():
                return
            with open(self.map_path, encoding="utf-8") as f:
                data = json.load(f)
            domains = data.get("domains") or {}
            loaded_dom = 0
            loaded_exp = 0
            for dom, v in domains.items():
                if not isinstance(v, dict):
                    continue
                per_layer = v.get("per_layer_domain_preferred") or {}
                for layer_str, experts in per_layer.items():
                    if not isinstance(experts, list):
                        continue
                    try:
                        layer = int(layer_str)
                    except ValueError:
                        continue
                    for eid in experts:
                        try:
                            eid = int(eid)
                        except (TypeError, ValueError):
                            continue
                        self._agg[dom][layer][eid] += 1  # 保底计数
                        loaded_exp += 1
                if per_layer:
                    loaded_dom += 1
            # 恢复领域热度榜（rebuild 写 dom_freq 用）
            old_freq = data.get("dom_freq") or {}
            for dom, n in old_freq.items():
                self._dom_freq[dom] += int(n)
            if loaded_dom:
                logger.info(
                    "Router flywheel: loaded existing map %s (%d domains, %d experts)",
                    self.map_path.name, loaded_dom, loaded_exp)
        except Exception as e:  # 加载失败不能崩，走全新收集
            logger.warning("Router flywheel: load existing map failed (fresh start): %s", e)

    # ── 入口 ────────────────────────────────────────────────

    def set_domain(self, domain: str) -> None:
        """请求级：当前请求判定为哪个领域（proxy 判完 domain 后调用）。

        [2026-08-13] 同时累计领域请求次数（热门领域热度榜）。
        """
        if not domain:
            return
        with self._lock:
            self._domain = domain
            self._dom_freq[domain] += 1

    def hot_domains(self, n: int) -> list[str]:
        """[2026-08-13] 热度榜前 N 名领域（按请求次数降序）。"""
        with self._lock:
            return [d for d, _ in self._dom_freq.most_common(n)]

    def on_expert_line(self, line: str) -> None:
        """喂一条 EXPERT 路由日志行（gate.on_log_line 同源解析，格式一致）。

        EXPERT|L0|T2: [64,161,...] [130,...]
        """
        import re

        m = re.match(r"EXPERT\|L(\d+)\|T\d+:\s*(.*)$", line.strip())
        if not m:
            return
        layer = int(m.group(1))
        body = m.group(2)
        with self._lock:
            dom_counter = self._agg[self._domain][layer]
            for seg in re.findall(r"\[([0-9, ]+)\]", body):
                for x in seg.split(","):
                    x = x.strip()
                    if x:
                        dom_counter[int(x)] += 1
                        self._records += 1
            if (self._records >= self.rebuild_every
                    and time.time() - self._last_rebuild >= self.rebuild_min_interval):
                self._rebuild_locked()

    def maybe_rebuild(self, force: bool = False) -> bool:
        """手动触发重建（或 force 强制）。返回是否重建了。"""
        with self._lock:
            if not force and self._records < self.min_records:
                logger.info(
                    "Router flywheel: %d records (< %d) — skip rebuild",
                    self._records, self.min_records)
                return False
            if force and self._records < self.min_records:
                logger.warning(
                    "Router flywheel: force rebuild with only %d records",
                    self._records)
            self._rebuild_locked()
            return True

    # ── 内部 ────────────────────────────────────────────────

    def _rebuild_locked(self) -> None:
        """用当前聚合计数重建路由表 JSON（原子替换）。调用方须持有锁。"""
        domains = {}
        for dom, layers in self._agg.items():
            per_layer = {}
            for layer in sorted(layers):
                counter = layers[layer]
                if not counter:
                    continue
                experts = [e for e, _ in counter.most_common(self.top_k)]
                per_layer[str(layer)] = experts
            if per_layer:
                domains[dom] = {"per_layer_domain_preferred": per_layer}

        if not domains:
            logger.warning("Router flywheel: no aggregated data, skip write")
            return

        out = {
            "description": (
                "Domain router map auto-built by flywheel from real gating "
                "traces ({} records). Rebuilt {} times. "
                "Per-domain per-layer top-{} experts.").format(
                    self._records, self._rebuilds, self.top_k),
            "source": "flywheel",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "top_k": self.top_k,
            "dom_freq": dict(self._dom_freq),  # [2026-08-13] 领域热度榜
            "domains": domains,
        }

        self.map_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.map_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.map_path)  # 原子替换
        self._rebuilds += 1
        self._records = 0
        self._last_rebuild = time.time()
        # 注意：不清空 _agg —— 数据飞轮要跨 rebuild 累积，
        # 每次重建都是"全量历史 + 新增"的聚合，路由表越用越准。
        logger.info(
            "Router flywheel: rebuilt %s (%d domains, top-%d)",
            self.map_path, len(domains), self.top_k)

    # ── 状态 ────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            return {
                "router_flywheel_records": self._records,
                "router_flywheel_rebuilds": self._rebuilds,
                "router_flywheel_domains": sorted(self._agg.keys()),
                "router_flywheel_rebuild_every": self.rebuild_every,
                "router_flywheel_top_k": self.top_k,
            }
