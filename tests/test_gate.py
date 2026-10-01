"""Tests for moe_l2.gate RoutingProfiler — 多领域保留池 v2（retain-hot-experts）。"""

import json
import os
import time
import urllib.request
from collections import deque
from pathlib import Path

from moe_l2.gate import (
    RoutingProfiler,
    _main_top_k_for_vram,
    _retain_params,
)

# 与 flywheel 表同构的假数据：3 领域 × 2 层
DOMAINS = {
    "codegen": {
        "per_layer_domain_preferred": {
            "0": [i for i in range(0, 40)],
            "1": [i for i in range(100, 140)],
        }
    },
    "debug": {
        "per_layer_domain_preferred": {
            "0": [i for i in range(50, 90)],
            "1": [i for i in range(150, 190)],
        }
    },
    "translate": {
        "per_layer_domain_preferred": {
            "0": [i for i in range(200, 240)],
            "1": [i for i in range(300, 340)],
        }
    },
}

DOM_FREQ = {"codegen": 10, "debug": 5, "translate": 2}


def _profiler(retain_top_k=30, pool_size=3, vram_mb=12288) -> RoutingProfiler:
    p = RoutingProfiler(retain_top_k=retain_top_k, pool_size=pool_size)
    p._router_top_k = 100
    return p


# ── 显存自适应档位 ─────────────────────────────────────────────

def test_main_top_k_vram_tiers():
    assert _main_top_k_for_vram(4096) == 20      # 4G
    assert _main_top_k_for_vram(6144) == 50      # 6G
    assert _main_top_k_for_vram(8192) == 75      # 8G
    assert _main_top_k_for_vram(11264) == 75     # 11GB 2080Ti → 8G 档
    assert _main_top_k_for_vram(12288) == 75     # 12G 边界仍 8G 档
    assert _main_top_k_for_vram(24576) == 100    # 24G → 12G+ 档
    assert _main_top_k_for_vram(0) == 75         # 探测失败兜底


def test_retain_params_vram_tiers():
    # 2026-08-17 拍板：默认单表运行（保留池关闭），任何显存都返回 (0, 1)
    assert _retain_params(4096) == (0, 1)
    assert _retain_params(6144) == (0, 1)
    assert _retain_params(11264) == (0, 1)
    assert _retain_params(24576) == (0, 1)
    assert _retain_params(0) == (0, 1)          # 探测失败兜底


def test_profiler_env_overrides(monkeypatch):
    monkeypatch.setenv("MOE_L2_MAIN_TOP_K", "50")
    monkeypatch.setenv("MOE_L2_RETAIN_TOP_K", "20")
    monkeypatch.setenv("MOE_L2_POOL_SIZE", "2")
    p = RoutingProfiler()
    assert p._router_top_k == 50
    assert p._retain_top_k == 20
    assert p._pool_size == 2


# ── 领域池构造 ─────────────────────────────────────────────────

def test_build_pool_current_plus_prev_plus_hot():
    p = _profiler()
    p._domain_history = deque(["debug", "codegen"])  # 最新 = codegen（上一个）
    pool = p._build_domain_pool("translate", DOMAINS, DOM_FREQ)
    # 当前 translate + 上一个 codegen + dom_freq 补位（debug 是热度第二）
    assert pool == ["translate", "codegen", "debug"]


def test_build_pool_hot_completes_missing_prev():
    p = _profiler()
    p._domain_history = deque(["codegen", "debug"])
    # 当前 = codegen（历史最新也是 codegen）→ 无上一个，池 = codegen + dom_freq 补位
    pool = p._build_domain_pool("codegen", DOMAINS, DOM_FREQ)
    assert pool == ["codegen", "debug", "translate"]


def test_build_pool_prev_equals_current_uses_hot():
    p = _profiler()
    p._domain_history = deque(["codegen", "codegen"])
    pool = p._build_domain_pool("codegen", DOMAINS, DOM_FREQ)
    # 上一个 == 当前 → 跳过，dom_freq 补位 debug（最高不在池的）
    assert pool == ["codegen", "debug", "translate"]


def test_build_pool_pool_size_cap():
    p = _profiler(pool_size=2)
    p._domain_history = deque(["debug", "codegen"])
    pool = p._build_domain_pool("translate", DOMAINS, DOM_FREQ)
    assert pool == ["translate", "codegen"]


def test_build_pool_skips_domains_not_in_table():
    p = _profiler()
    p._domain_history = deque(["translate", "codegen"])
    dom_freq = {"math": 999, "codegen": 10}  # math 不在表内
    pool = p._build_domain_pool("debug", DOMAINS, dom_freq)
    # math 应被跳过 → 池 = debug + codegen（无第三个候选）
    assert pool == ["debug", "codegen"]


def test_build_pool_empty_history():
    p = _profiler()
    pool = p._build_domain_pool("codegen", DOMAINS, {})
    assert pool == ["codegen"]


# ── retain payload 构造 ─────────────────────────────────────────

def test_build_retain_payload_union_of_other_domains():
    p = _profiler(retain_top_k=10)
    pool = ["translate", "codegen", "debug"]  # 主表 = translate
    retain = p._build_retain_payload("translate", DOMAINS, pool)
    assert retain is not None
    # 每层 = codegen top-10 ∪ debug top-10（translate 自己不进 retain）
    assert retain["0"] == sorted(list(range(0, 10)) + list(range(50, 60)))
    assert retain["1"] == sorted(list(range(100, 110)) + list(range(150, 160)))


def test_build_retain_payload_pool_only_main():
    p = _profiler()
    retain = p._build_retain_payload("codegen", DOMAINS, ["codegen"])
    assert retain is None


def test_build_retain_payload_respects_retain_top_k():
    p = _profiler(retain_top_k=5)
    pool = ["translate", "codegen"]
    retain = p._build_retain_payload("translate", DOMAINS, pool)
    assert len(retain["0"]) == 5
    assert retain["0"] == list(range(0, 5))


# ── _switch_router_domain 完整链路（mock urlopen） ───────────────

class _FakeResp:
    def read(self):
        return b"{}"


def _patch_urlopen(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["url"] = req.full_url
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return captured


def _write_map(tmp_path: Path) -> Path:
    p = tmp_path / "flywheel.json"
    p.write_text(json.dumps({
        "dom_freq": DOM_FREQ,
        "domains": DOMAINS,
    }), encoding="utf-8")
    return p


class _FakeFlywheel:
    def __init__(self, map_path):
        self.map_path = map_path


def test_switch_domain_sends_retain(tmp_path, monkeypatch):
    captured = _patch_urlopen(monkeypatch)
    p = _profiler()
    p.router_server_url = "http://127.0.0.1:8080"
    p.router_flywheel = _FakeFlywheel(_write_map(tmp_path))
    p._domain_history = deque(["codegen", "debug"])

    p._switch_router_domain("translate")

    body = captured["body"]
    assert captured["url"] == "http://127.0.0.1:8080/moe-set-domain"
    assert body["domain"] == "translate"
    # 主表 = translate 每层 top-100（表内 40 个全取）
    assert len(body["experts"]["0"]) == 40
    # retain = codegen + debug 每层 top-30
    assert len(body["retain"]["0"]) == 60
    assert len(body["retain"]["1"]) == 60


def test_switch_domain_history_updated(tmp_path, monkeypatch):
    _patch_urlopen(monkeypatch)
    p = _profiler()
    p.router_server_url = "http://127.0.0.1:8080"
    p.router_flywheel = _FakeFlywheel(_write_map(tmp_path))

    p._switch_router_domain("codegen")
    assert list(p._domain_history) == ["codegen"]
    p._switch_router_domain("debug")
    assert list(p._domain_history) == ["codegen", "debug"]


def test_switch_domain_no_map_skips(tmp_path, monkeypatch):
    captured = _patch_urlopen(monkeypatch)
    p = _profiler()
    p.router_server_url = "http://127.0.0.1:8080"
    p.router_flywheel = _FakeFlywheel(tmp_path / "missing.json")
    p._switch_router_domain("codegen")
    assert "body" not in captured


def test_switch_domain_domain_not_in_table(tmp_path, monkeypatch):
    captured = _patch_urlopen(monkeypatch)
    p = _profiler()
    p.router_server_url = "http://127.0.0.1:8080"
    p.router_flywheel = _FakeFlywheel(_write_map(tmp_path))
    p._switch_router_domain("math")  # 不在表里 → skip
    assert "body" not in captured


# ── [走法 D 2026-09-19] 表生成后主动热换（免重启） ─────────────────

def _wait_until(cond, timeout=10.0):
    # 10s（原 3s）：推表在后台线程里跑（urlopen 已 mock，但线程调度受全量跑负载影响），
    # 3s 在整套测试并发跑时偶发超时 → test_table_push_on_update 间歇性失败（时序 flaky）。
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


def _d_profiler(tmp_path, monkeypatch):
    """假飞轮 + 节流关掉的 profiler，urlopen 已 mock。"""
    captured = _patch_urlopen(monkeypatch)
    p = _profiler()
    p.router_server_url = "http://127.0.0.1:8080"
    p.router_flywheel = _FakeFlywheel(tmp_path / "flywheel.json")
    p._table_check_interval = 0.0
    # 真实运行时 flywheel 是构造参数（__init__ 里就登记了表基线）；测试里是后挂
    # 上去的，这里补一次登记，等价于"进程启动那一刻"的表状态。
    p._table_sig = p._current_table_sig()
    p._table_push_enabled = True   # 2026-10-01 起默认即为 True；测试里显式写清意图
    return p, captured


def test_table_push_disabled_without_flywheel_or_url():
    p = _profiler()
    assert p._maybe_switch_on_new_table() is False


def test_table_push_first_check_only_records_baseline(tmp_path, monkeypatch):
    """无表启动 + 首次检查：只登记基线，不推送。"""
    p, captured = _d_profiler(tmp_path, monkeypatch)
    assert p._maybe_switch_on_new_table() is False
    assert "body" not in captured


def test_table_push_when_table_appears(tmp_path, monkeypatch):
    """真表出现 → 主动推一次（当前领域不在表里 → 取 dom_freq 最热）。"""
    p, captured = _d_profiler(tmp_path, monkeypatch)
    p._maybe_switch_on_new_table()          # 基线：无表
    _write_map(tmp_path)                    # 飞轮写出第一张表
    assert p._maybe_switch_on_new_table() is True
    assert _wait_until(lambda: "body" in captured)
    assert captured["url"] == "http://127.0.0.1:8080/moe-set-domain"
    assert captured["body"]["domain"] == "codegen"      # DOM_FREQ 最高
    assert len(captured["body"]["experts"]["0"]) == 40  # 主表 top-100（表内 40 全取）
    assert _wait_until(lambda: p._table_pushes == 1)


def test_table_push_no_repeat_when_unchanged(tmp_path, monkeypatch):
    """启动时已有表：不推；文件没变也不推（启动流程已经加载过它）。"""
    _write_map(tmp_path)
    p, captured = _d_profiler(tmp_path, monkeypatch)
    assert p._maybe_switch_on_new_table() is False
    assert p._maybe_switch_on_new_table() is False
    assert "body" not in captured


def test_table_push_when_table_appears_before_first_check(tmp_path, monkeypatch):
    """表在"启动之后、第一个请求之前"出现（后台采集/手动 collect）也必须推。

    这是 2026-09-19 210 实测抓出来的 bug：基线若等首次检查才登记，这种表会
    被当成基线、永久漏推（当时靠请求级领域切换兜住了，掩盖了问题）。
    """
    p, captured = _d_profiler(tmp_path, monkeypatch)   # __init__ 基线：无表
    _write_map(tmp_path)
    assert p._maybe_switch_on_new_table() is True
    assert _wait_until(lambda: p._table_pushes == 1)


def test_table_push_on_update(tmp_path, monkeypatch):
    """表被重建（文件变了）→ 再推一次。"""
    p, captured = _d_profiler(tmp_path, monkeypatch)
    p._maybe_switch_on_new_table()          # 基线
    path = _write_map(tmp_path)
    assert p._maybe_switch_on_new_table() is True
    assert _wait_until(lambda: p._table_pushes == 1)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["dom_freq"] = {"debug": 99, **DOM_FREQ}
    path.write_text(json.dumps(data), encoding="utf-8")
    assert p._maybe_switch_on_new_table() is True
    assert _wait_until(lambda: p._table_pushes == 2)
    assert captured["body"]["domain"] == "codegen"  # 当前领域为空 → 仍按 dom_freq


def test_table_push_skips_when_table_has_no_domains(tmp_path, monkeypatch):
    p, captured = _d_profiler(tmp_path, monkeypatch)
    p._maybe_switch_on_new_table()
    (tmp_path / "flywheel.json").write_text(
        json.dumps({"domains": {}}), encoding="utf-8")
    assert p._maybe_switch_on_new_table() is False
    assert "body" not in captured


def test_table_push_prefers_current_domain(tmp_path, monkeypatch):
    """飞轮当前领域在表里 → 优先推它（而不是最热的那个）。"""
    p, captured = _d_profiler(tmp_path, monkeypatch)
    p.router_flywheel._domain = "translate"
    p._maybe_switch_on_new_table()
    _write_map(tmp_path)
    assert p._maybe_switch_on_new_table() is True
    assert _wait_until(lambda: p._table_pushes == 1)
    assert captured["body"]["domain"] == "translate"


def test_table_watcher_pushes_without_log_lines(tmp_path, monkeypatch):
    """日志静默（没有请求）时表出现也要推 —— 独立看门线程。

    210 实测踩到的场景：真表是在服务空闲时出现的（手动 collect / 后台采集），
    只靠 EXPERT 日志行回调永远检测不到。
    """
    p, captured = _d_profiler(tmp_path, monkeypatch)
    p._table_check_interval = 0.05
    th = p.start_table_watcher(interval=0.05)
    assert th is not None
    try:
        _write_map(tmp_path)                      # 没有任何请求 / 日志行
        assert _wait_until(lambda: p._table_pushes == 1, timeout=5.0)
    finally:
        p.stop_table_watcher()


def test_table_watcher_disabled_without_flywheel_or_url():
    assert _profiler().start_table_watcher() is None


def test_table_push_deferred_while_engine_busy(tmp_path, monkeypatch):
    """引擎在生成中（日志还热）时不换表 —— 只在空闲时换。

    210 实测：占位表 + 运行中换表 + 连续请求出现过 llama-server 崩溃
    （ggml-base.dll / 0xc0000005），这条保护把换表推到日志静默之后。
    """
    import time as _time
    p, captured = _d_profiler(tmp_path, monkeypatch)
    _write_map(tmp_path)
    p._last_log_ts = _time.time()          # 引擎刚还在吐 token
    assert p._maybe_switch_on_new_table() is False
    assert "body" not in captured
    # 日志静下来（引擎空闲）→ 允许换
    p._last_log_ts = _time.time() - 60.0
    assert p._maybe_switch_on_new_table() is True
    assert _wait_until(lambda: p._table_pushes == 1)


def test_table_push_and_watcher_enabled_by_default(monkeypatch):
    """自动推表默认**开启**（2026-10-01 起：C++ 换表竞态已修，补丁版实测
    Windows 110 次 / Linux 40 次生成中途换表零崩）；MOE_L2_TABLE_PUSH=0 可关。"""
    monkeypatch.delenv("MOE_L2_TABLE_PUSH", raising=False)
    p = RoutingProfiler(retain_top_k=30, pool_size=3)
    assert p._table_push_enabled is True
    monkeypatch.setenv("MOE_L2_TABLE_PUSH", "0")
    p_off = RoutingProfiler(retain_top_k=30, pool_size=3)
    assert p_off._table_push_enabled is False
    monkeypatch.setenv("MOE_L2_TABLE_PUSH", "1")
    p_on = RoutingProfiler(retain_top_k=30, pool_size=3)
    assert p_on._table_push_enabled is True


def test_idle_threshold_default_is_half_second(monkeypatch):
    monkeypatch.delenv("MOE_L2_SWAP_IDLE_SEC", raising=False)
    p = RoutingProfiler()
    assert p._idle_before_switch == 0.5
