"""门控在线自适应（P2 ④，2026-08-02）。

推理中读 llama-server 的 LLAMA_EXPERT_LOG 实时路由输出，动态调整
L2 缓存优先级（LRU 智能增强），并按会话累积路由画像。

与分类器的分工（产品化方案定稿）：
- 分类器（L0a）管冷启动：prompt → 领域，预测该预载哪些专家
- 门控管热循环：推理中 → 真实路由信号 → 动态调缓存优先级

信号流：
    llama-server stderr (LLAMA_EXPERT_LOG=1)
        → EXPERT|L3|T4: [12,45] [3,77] ...
        → RoutingProfiler.on_log_line(line)  解析 + 更新画像
        → 窗口内漂移检测 → cache.promote_domain() / promote 高频专家
        → 缓存优先级自适应

用法：
    from moe_l2.gate import RoutingProfiler
    profiler = RoutingProfiler(cache, expert_map)
    for line in server_stderr_lines:
        profiler.on_log_line(line)      # 实时路由 → 调缓存
    profiler.on_request(domain)         # 请求级信号 → 检测漂移
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.request
from collections import defaultdict, deque

logger = logging.getLogger("moe-l2-gate")

# EXPERT|L3|T4: [12,45] [3,77] ...
_EXPERT_LINE_RE = re.compile(r"EXPERT\|L(\d+)\|T(\d+): (.*)")

# 滑动窗口：保留最近 N 层-行数的激活记录做漂移检测
_WINDOW_LINES = 400

# [moe-l2 retain-hot-experts v2 2026-08-16] 多领域保留池默认参数
# 主表 top-K 显存自适应档位（Qwen 实测口径，v2 方案第三节档位表）：
#   ≤4G: 主表 20 + 保留 10 + 池 2；≤6G: 50+30+2；≤12G: 75+30+3（2080Ti 11GB 实测 7.7GB）；
#   >12G: 100+30+3（12G+ 需 MAX_SLOTS 提升至 20400 槽，见待办）
_RETAIN_TOP_K_DEFAULT = 30
_POOL_SIZE_DEFAULT = 3
# 主表 top-K 按显存自适应（MB → top-k），与 router_table COVERAGE_TABLE 对齐
_MAIN_TOP_K_VRAM = (
    (4096, 20),
    (6144, 50),
    (12288, 75),  # 8G 档覆盖到 12G（2080Ti 11GB 实测走此档）
    (float("inf"), 100),
)


def _main_top_k_for_vram(vram_total_mb: int) -> int:
    """按显存总量（MB）返回主表 top-K 档位。探测失败（0）按 8G 档 75。"""
    if vram_total_mb <= 0:
        return 75
    for limit, k in _MAIN_TOP_K_VRAM:
        if vram_total_mb <= limit:
            return k
    return 100


def _retain_params(vram_total_mb: int) -> tuple[int, int]:
    """按显存返回 (retain_top_k, pool_size)。

    [moe-l2 retain-hot-experts v2 2026-08-17 拍板：默认单表运行]
    保留池实测结论（4090，2026-08-17）：DS 换表 payload 变大（主表+3领域并集）
    导致净亏 ~20%（89-103 vs 单表 116-127 t/s）；Qwen 混合场景也 -7~12%，
    回切收益（+20.5%）不足以抵消 → 默认关闭保留池（pool_size=1、retain=0），
    行为等同 v0.5.0 C 方案单表。环境变量 MOE_L2_RETAIN_TOP_K / MOE_L2_POOL_SIZE
    仍可临时开启（如 MOE_L2_POOL_SIZE=3）。
    """
    return 0, 1


def _detect_vram_mb() -> int:
    """探测 GPU 总显存（MB）。nvidia-smi 失败/无卡返回 0（按 8G 档兜底）。"""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if out:
            return int(float(out.splitlines()[0]))
    except Exception:
        pass
    return 0


class RoutingProfiler:
    """会话路由画像 + 在线自适应门控。

    - 画像：每层专家激活频率（滑动窗口，反映"最近在用谁"）
    - 漂移检测：当前窗口与基线窗口的 Jaccard 差异 → 路由漂移信号
    - 动作：漂移 → promote 目标域；高频专家 → 自动抬优先级
    """

    def __init__(
        self,
        cache=None,
        expert_map: dict | None = None,
        drift_threshold: float = 0.35,
        router_flywheel=None,
        router_server_url: str | None = None,
        retain_top_k: int | None = None,
        pool_size: int | None = None,
    ):
        self.cache = cache
        self.expert_map = expert_map or {}
        self.drift_threshold = drift_threshold
        # [moe-l2 2026-08-09] 路由表数据飞轮（可选，None = 不启用，向后兼容）
        self.router_flywheel = router_flywheel
        # [moe-l2 route-by-domain C 2026-08-15] llama-server 地址；设置后
        # on_request(domain) 会把该领域的专家表 POST /moe-set-domain（动态换表）。
        self.router_server_url = router_server_url
        self._last_router_switch = 0.0

        # [moe-l2 retain-hot-experts v2 2026-08-16] 多领域保留池参数。
        # 主表 top-K + 保留 top-X + 池大小按显存自适应（v2 方案第三节档位表）。
        # 显式参数 > 环境变量 > 显存自适应默认。
        vram_mb = _detect_vram_mb()
        main_top_k = int(os.environ.get(
            "MOE_L2_MAIN_TOP_K", _main_top_k_for_vram(vram_mb)))
        if retain_top_k is None:
            retain_top_k = int(os.environ.get(
                "MOE_L2_RETAIN_TOP_K", _retain_params(vram_mb)[0]))
        if pool_size is None:
            pool_size = int(os.environ.get(
                "MOE_L2_POOL_SIZE", _retain_params(vram_mb)[1]))
        # 每层最多保留的专家数（主表，与 build_router_map_file top-k 对齐）
        self._router_top_k = main_top_k
        # 保留池：每领域每层保留的专家数 + 池大小
        self._retain_top_k = retain_top_k
        self._pool_size = pool_size
        # 领域历史队列（当前 + 上一个，用于构造保留池）
        self._domain_history: deque[str] = deque(maxlen=2)
        logger.info(
            "Gate: retain pool v2 vram=%dMB main_top_k=%d retain_top_k=%d pool_size=%d",
            vram_mb, self._router_top_k, self._retain_top_k, self._pool_size)

        # 窗口内激活记录（FIFO，用于漂移对比）
        self._window: deque[tuple[int, int]] = deque(maxlen=_WINDOW_LINES)

        # 会话画像：layer -> Counter(expert_id -> 激活次数)
        self._profile: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # 上次漂移检测时的窗口快照（对比基线）
        self._baseline: set[tuple[int, int]] = set()

        self._log_lines = 0
        self._last_drift: float = 0.0

        # [moe-l2 2026-08-19 proxy 并发卡死修复] 换表节流 + 互斥 + 后台线程。
        # 根因：旧版 on_request 每次请求都同步 POST /moe-set-domain（timeout=10s），
        # llama-server 端清槽 + 重 prefill 120 条耗时数秒 → 并发请求全部阻塞在
        # 换表上排队卡死（2080Ti 实测：并发测试基线请求后无响应）。
        # 修复：① 同一领域 30s 内不重复换表（旧 _last_router_switch 从未使用，
        # 节流从未生效）；② 全局互斥，已有换表在跑则跳过（请求转发用当前表，
        # prefill 未换只影响预热不影响正确性）；③ 换表放后台 daemon 线程，
        # 请求线程不被阻塞。
        self._router_switch_lock = threading.Lock()
        self._router_last_switch: dict[str, float] = {}

        # [走法 D 2026-09-19] 飞轮表文件监测：真表在运行中才写出来（启动时可能
        # 只有占位表），而引擎不会自己重读文件 → 这里盯表文件的 (mtime, size)，
        # 一变就主动 POST /moe-set-domain 热换。用户不必重启、也不必切领域。
        # 基线在 __init__ 就登记（不是等首次检查）：表若在"启动之后、第一个请求
        # 之前"出现（后台采集 / 手动 collect / 飞轮写出），也要能被认成变化。
        self._table_sig: tuple[float, int] | None = self._current_table_sig()
        self._last_table_check = 0.0
        self._table_pushes = 0
        # 空闲阈值：生成期每 ~50ms 就有一条日志，0.5s 足够区分"正在生成"和
        # "请求间隙"（2.0s 太保守：1s 间隔的连续请求永远推不进去，210 实测）。
        try:
            self._idle_before_switch = float(
                os.environ.get("MOE_L2_SWAP_IDLE_SEC", "0.5"))
        except ValueError:
            self._idle_before_switch = 0.5
        # [2026-09-19 安全默认] 表更新自动推送默认关闭。原因：换表路径在 C++ 侧
        # 有并发缺陷（clear()/set_layer() 直接改全局 g_router_map，而推理线程同时
        # 读它 → 越界访问 0xc0000005；210 实测 20 轮内复现，12 轮不换表则稳定）。
        # 飞轮修活后推送频率上升会显著提高触发概率，因此先关；等 C++ 改成
        # "构建新表 → 原子替换" 后再打开：MOE_L2_TABLE_PUSH=1。
        self._table_push_enabled = os.environ.get(
            "MOE_L2_TABLE_PUSH", "").strip().lower() in ("1", "true", "on", "yes")
        self._table_watch_thread: threading.Thread | None = None
        self._table_watch_stop: threading.Event | None = None
        # [2026-09-19 保护] 最近一条引擎日志的时刻：生成期间每 token 40 行日志，
        # 日志静下来才说明引擎空闲 → 只在这时换表，避开"生成中途 clear/set_layer"
        # 交错（210 实测：占位表 + 运行中换表 + 连续请求出现过 ggml-base.dll
        # 0xc0000005 崩溃）。
        self._last_log_ts = 0.0
        try:
            self._table_check_interval = float(
                os.environ.get("MOE_L2_TABLE_CHECK_SEC", "10"))
        except ValueError:
            self._table_check_interval = 10.0

    def _current_table_sig(self) -> tuple[float, int] | None:
        """飞轮表文件的 (mtime, size)；不存在/读不到 → None。"""
        if self.router_flywheel is None:
            return None
        map_path = getattr(self.router_flywheel, "map_path", None)
        if map_path is None:
            return None
        try:
            st = os.stat(map_path)
        except OSError:
            return None
        return (st.st_mtime, st.st_size)

    def _maybe_switch_on_new_table(self) -> bool:
        """飞轮表文件出现/更新 → 主动热换一次（后台线程，不阻塞日志线程）。

        成本：一次 os.stat（按 _table_check_interval 节流，默认 10s）。
        返回 True 表示本轮触发了推送。
        """
        if self.router_flywheel is None or not self.router_server_url:
            return False
        now = time.time()
        if now - self._last_table_check < self._table_check_interval:
            return False
        self._last_table_check = now

        sig = self._current_table_sig()
        if sig is None or sig == self._table_sig:
            return False
        # 引擎忙 → 推迟（表签名先不更新，下个周期再来）
        idle_for = now - self._last_log_ts
        if idle_for < self._idle_before_switch:
            logger.info("Gate: engine busy (log %.1fs ago), defer table swap", idle_for)
            return False
        self._table_sig = sig

        map_path = getattr(self.router_flywheel, "map_path", None)
        domain = self._pick_domain_for_push(map_path)
        if domain is None:
            logger.info("Gate: new router table has no usable domain, skip push")
            return False
        if not self._router_switch_lock.acquire(blocking=False):
            return False  # 已有换表在跑 → 下次再推（表文件签名已更新，不会漏）

        def _do() -> None:
            try:
                self._switch_router_domain(domain)
                self._table_pushes += 1
                logger.info(
                    "Gate: hot-swapped router table (domain=%s, push #%d, no restart)",
                    domain, self._table_pushes)
            except Exception as e:  # noqa: BLE001 - 后台线程兜底
                logger.warning("Gate table push failed (non-fatal): %s", e)
            finally:
                self._router_switch_lock.release()

        threading.Thread(target=_do, daemon=True,
                         name="moe-l2-table-push").start()
        return True

    def start_table_watcher(self, interval: float | None = None):
        """[走法 D] 独立后台线程：周期盯飞轮表文件，出现/更新就热换。

        为什么不只靠日志行回调（210 实测踩到）：① 日志流只在有请求时才动，服务
        空闲时出现真表（手动 collect / 后台采集）永远检测不到；② 检查点若正好
        撞上领域换表的锁（请求刚开始时），那一次会被跳过，而请求结束后日志静默
        → 再没有重试机会。独立线程 + 周期检查把这两个口子一起堵上。
        """
        if self.router_flywheel is None or not self.router_server_url:
            return None
        if not self._table_push_enabled:
            logger.info("Gate: table auto-push disabled "
                        "(set MOE_L2_TABLE_PUSH=1 to enable)")
            return None
        if self._table_watch_thread is not None:
            return self._table_watch_thread
        if interval is None:
            interval = min(self._table_check_interval, 5.0) or 5.0
        interval = float(interval)
        self._table_watch_stop = threading.Event()

        def _loop() -> None:
            stop = self._table_watch_stop
            while stop is not None and not stop.wait(interval):
                try:
                    self._maybe_switch_on_new_table()
                except Exception as e:  # noqa: BLE001 - 后台线程兜底
                    logger.warning("Gate table watcher failed (non-fatal): %s", e)

        self._table_watch_thread = threading.Thread(
            target=_loop, daemon=True, name="moe-l2-table-watcher")
        self._table_watch_thread.start()
        logger.info("Gate: router-table watcher started (every %.0fs)", interval)
        return self._table_watch_thread

    def stop_table_watcher(self) -> None:
        if self._table_watch_stop is not None:
            self._table_watch_stop.set()

    def _pick_domain_for_push(self, map_path) -> str | None:
        """选一个表里确实有专家的领域来推：当前领域 → dom_freq 最热 → 任意。"""
        try:
            with open(map_path, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:  # noqa: BLE001 - 表刚写一半/损坏都当没表
            logger.info("Gate: cannot read new router table (%s), skip push", e)
            return None
        domains = d.get("domains", {})
        if not domains:
            return None
        cur = str(getattr(self.router_flywheel, "_domain", "") or "")
        if cur and cur in domains:
            return cur
        dom_freq = d.get("dom_freq", {}) or {}
        for dom in sorted(dom_freq, key=lambda k: -dom_freq.get(k, 0)):
            if dom in domains:
                return dom
        return sorted(domains)[0]

    def _schedule_router_switch(self, domain: str) -> None:
        """节流 + 互斥 + 后台执行动态换表（不阻塞请求线程）。"""
        now = time.time()
        if now - self._router_last_switch.get(domain, 0.0) < 30.0:
            return
        if not self._router_switch_lock.acquire(blocking=False):
            return  # 已有换表在跑 → 跳过本次（不排队，避免堆积）
        self._router_last_switch[domain] = now

        def _do() -> None:
            try:
                self._switch_router_domain(domain)
            except Exception as e:  # noqa: BLE001 - 后台线程兜底
                logger.warning("Gate router switch thread failed (non-fatal): %s", e)
            finally:
                self._router_switch_lock.release()

        threading.Thread(target=_do, daemon=True,
                         name="moe-l2-router-switch").start()

    # ── 实时信号入口 ──────────────────────────────────────────

    def on_log_line(self, line: str) -> None:
        """解析一条 EXPERT 日志行并更新画像 / 触发动作。"""
        m = _EXPERT_LINE_RE.match(line.strip())
        if not m:
            return
        self._last_log_ts = time.time()  # 引擎在干活（换表保护用）
        layer = int(m.group(1))
        body = m.group(3)
        self._log_lines += 1

        expert_ids: list[int] = []
        for seg in re.findall(r"\[([0-9, ]+)\]", body):
            for x in seg.split(","):
                x = x.strip()
                if x:
                    expert_ids.append(int(x))

        for eid in expert_ids:
            self._window.append((layer, eid))
            self._profile[layer][eid] += 1

        # 每 40 行做一次漂移检测（避免每行都算）
        if self._log_lines % 40 == 0:
            self._maybe_promote()
            self._detect_drift()
            # [走法 D 2026-09-19] 顺带盯一眼飞轮表文件（内部按 10s 节流 + os.stat）
            self._maybe_switch_on_new_table()

        # [moe-l2 2026-08-09] 路由表数据飞轮：真实路由 → 按领域聚合
        if self.router_flywheel is not None:
            try:
                self.router_flywheel.on_expert_line(line)
            except Exception as e:
                logger.warning("Router flywheel line failed (non-fatal): %s", e)

    def on_request(self, domain: str) -> None:
        """请求级信号：领域切换 → 主动预热目标域。"""
        # [moe-l2 2026-08-16] domain 归一化为 str：predict_hybrid 可能返回
        # np.str_（numpy 字符串），混入历史队列/池会污染日志与序列化。
        domain = str(domain)
        # [moe-l2 2026-08-09] 先同步领域给飞轮（聚合按当前领域归组）
        if self.router_flywheel is not None:
            try:
                self.router_flywheel.set_domain(domain)
            except Exception as e:
                logger.warning("Router flywheel set_domain failed (non-fatal): %s", e)
        # [moe-l2 route-by-domain C 2026-08-15] 动态换表：把该领域的专家表
        # POST /moe-set-domain → llama-server 清旧 prefill、按新表重新 prefill。
        # 放在 cache 判断之前：即使无 L2 cache 也能吃到领域专属 prefill 加速。
        # [moe-l2 2026-08-19] 改为节流+互斥+后台执行：旧版同步换表在并发请求下
        # 全部阻塞在 /moe-set-domain（timeout=10s）→ proxy 并发卡死。
        if self.router_server_url is not None:
            self._schedule_router_switch(domain)
        if self.cache is None or self.expert_map is None:
            return
        try:
            self.cache.promote_domain(domain, self.expert_map)
            logger.info("Gate: request domain=%s → promote_domain", domain)
        except Exception as e:
            logger.warning("Gate promote failed: %s", e)

    def _switch_router_domain(self, domain: str) -> None:
        """按预测领域动态换路由表 + 保留上领域热专家（C 方案 + retain v2）。

        [moe-l2 retain-hot-experts v2 2026-08-16] 多领域保留池：
        - 领域历史队列（当前 + 上一个）→ 构造保留池
        - 池 = {当前} ∪ {上一个} ∪ {flywheel dom_freq 热门补位}（去重，≤ pool_size）
        - retain payload = 池中除主表外所有领域 top-X 并集（每层）
        - POST /moe-set-domain 带 retain：server 端主表 ∪ 保留按层合并去重 →
          soft_resize（不清 cache）→ 领域回切时旧领域热专家直接命中。

        失败非致命，只记日志（服务不中断）。
        """
        try:
            if self.router_flywheel is None:
                return
            map_path = getattr(self.router_flywheel, "map_path", None)
            if map_path is None or not map_path.exists():
                logger.info("Gate: no flywheel map %s, skip router switch", map_path)
                return
            with open(map_path, encoding="utf-8") as f:
                d = json.load(f)
            domains = d.get("domains", {})
            domdata = domains.get(domain)
            if not domdata:
                # 领域不在表里：退化为全领域并集（现有表内容），仅重置 prefill
                logger.info("Gate: domain %s not in flywheel table, skip switch", domain)
                return
            plist = domdata.get("per_layer_domain_preferred", {})
            k = self._router_top_k
            experts = {
                str(L): [int(e) for e in list(v)[:k]]
                for L, v in sorted(plist.items(), key=lambda kv: int(kv[0]))
            }
            if not experts:
                return

            # [moe-l2 retain-hot-experts v2 2026-08-16] 领域历史 + 池构造
            self._domain_history.append(domain)
            dom_freq = d.get("dom_freq", {})
            pool = self._build_domain_pool(domain, domains, dom_freq)
            retain = self._build_retain_payload(domain, domains, pool)

            body = {"domain": domain, "experts": experts}
            if retain:
                body["retain"] = retain
            body_b = json.dumps(body).encode("utf-8")
            base = self.router_server_url
            if not base:
                return
            url = base.rstrip("/") + "/moe-set-domain"
            req = urllib.request.Request(url, data=body_b,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
            logger.info(
                "Gate: router switch domain=%s layers=%d experts=%d retain=%s pool=%s",
                domain, len(experts), sum(len(v) for v in experts.values()),
                bool(retain), pool)
        except Exception as e:
            logger.warning("Gate router switch failed (non-fatal): %s", e)

    # [moe-l2 retain-hot-experts v2 2026-08-16] ── 多领域保留池 ────────────

    def _build_domain_pool(self, domain: str, domains: dict,
                           dom_freq: dict) -> list[str]:
        """构造领域池：{当前} ∪ {上一个} ∪ {dom_freq 热门补位}，去重至 pool_size。

        池内领域都必须存在于 flywheel 表（否则 retain 取不到专家）。
        热门 = dom_freq 降序取，跳过已在池的，直到池满或候选耗尽。
        """
        pool: list[str] = []
        # 1. 当前领域（主表）
        if domain in domains:
            pool.append(domain)
        # 2. 上一个领域（历史队列中最近的非当前领域）
        for prev in reversed(self._domain_history):
            if prev != domain and prev in domains and prev not in pool:
                pool.append(prev)
                break
        # 3. dom_freq 热门补位（跳过已在池的，保持池满 pool_size）
        ranked = [x for x, _ in sorted(dom_freq.items(), key=lambda kv: -kv[1])
                  if x in domains]
        for x in ranked:
            if len(pool) >= self._pool_size:
                break
            if x not in pool:
                pool.append(x)
        return pool[:self._pool_size]

    def _build_retain_payload(self, domain: str, domains: dict,
                              pool: list[str]) -> dict | None:
        """retain payload：池中除主表外所有领域 top-X 并集（每层）。

        返回 {layer_str: [expert...]}；池只有主表时返回 None（不传 retain）。
        """
        retain_layers: dict[int, set] = defaultdict(set)
        for rd in pool:
            if rd == domain:
                continue
            domdata = domains.get(rd)
            if not domdata:
                continue
            plist = domdata.get("per_layer_domain_preferred", {})
            for L, v in plist.items():
                retain_layers[int(L)].update(int(e) for e in list(v)[:self._retain_top_k])
        if not retain_layers:
            return None
        return {str(L): sorted(es) for L, es in sorted(retain_layers.items())}

    # ── 画像查询 ─────────────────────────────────────────────

    def top_experts(self, layer: int, k: int = 8) -> list[int]:
        """当前画像中该层激活最频繁的专家。"""
        cnt = self._profile.get(layer, {})
        return [e for e, _ in sorted(cnt.items(), key=lambda kv: -kv[1])[:k]]

    def profile_size(self) -> int:
        return sum(len(v) for v in self._profile.values())

    # ── 内部：在线自适应 ─────────────────────────────────────

    def _maybe_promote(self) -> None:
        """高频专家自动抬优先级（不 pin，靠 LRU 保留）。"""
        if self.cache is None:
            return
        # 取画像中每层 top-4 高频专家，promote（抬 MRU）
        with self.cache._lock:
            for layer, cnt in self._profile.items():
                top = [e for e, _ in sorted(cnt.items(), key=lambda kv: -kv[1])[:4]]
                for eid in top:
                    if self.cache._is_cached(layer, eid):
                        self.cache._touch(layer, eid)

    def _detect_drift(self) -> None:
        """检测路由漂移：当前窗口 vs 基线窗口的 Jaccard 差异。"""
        current = set(self._window)
        if not self._baseline or not current:
            # 首个窗口：建立基线
            self._baseline = current
            return

        inter = len(current & self._baseline)
        union = len(current | self._baseline)
        jaccard = inter / union if union else 1.0
        drift = 1.0 - jaccard  # 1 = 完全不同（漂移大），0 = 完全相同

        self._last_drift = drift
        self._baseline = current

        if drift >= self.drift_threshold:
            logger.info("Gate: routing drift %.2f detected (threshold %.2f)", drift, self.drift_threshold)
            # 漂移 → 主动把当前高频专家抬优先级（相当于"换挡"）
            self._maybe_promote()

    @property
    def last_drift(self) -> float:
        return self._last_drift

    def stats(self) -> dict:
        return {
            "gate_log_lines": self._log_lines,
            "gate_profile_experts": self.profile_size(),
            "gate_last_drift": round(self._last_drift, 3),
            "gate_drift_threshold": self.drift_threshold,
        }
