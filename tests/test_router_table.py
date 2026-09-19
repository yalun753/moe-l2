"""Tests for moe_l2.router_table (per-model router table selection + auto collect)."""

import json
import os

import pytest

from moe_l2.router_table import (
    _parse_router_map_json,
    build_router_map_file,
    compute_budget,
    filter_hot_domains,
    find_router_table,
    model_id_from_path,
    top_k_for_coverage,
)

# ── model_id_from_path ─────────────────────────────────────────────

def test_model_id_from_path_strips_gguf_and_dots():
    assert model_id_from_path("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf") == "Qwen3-6-35B-A3B-UD-IQ2_M"
    assert model_id_from_path("ds.gguf") == "ds"


# ── _parse_router_map_json ─────────────────────────────────────────

def test_parse_new_format_layers_key(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"layers": {"0": [1, 2, 3], "1": [4]}}))
    assert _parse_router_map_json(p) == {0: [1, 2, 3], 1: [4]}


def test_parse_new_format_numeric_keys(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"0": [10, 20], "3": [30]}))
    assert _parse_router_map_json(p) == {0: [10, 20], 3: [30]}


def test_parse_old_format_domains_union(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({
        "domains": {
            "codegen": {"per_layer_domain_preferred": {"0": [1, 2], "1": [5]}},
            "qa": {"per_layer_domain_preferred": {"0": [2, 3]}},
        }
    }))
    # layer 0 = union of both domains, sorted; layer 1 = only codegen
    assert _parse_router_map_json(p) == {0: [1, 2, 3], 1: [5]}


# ── find_router_table ──────────────────────────────────────────────

def test_find_router_table_prefers_per_model_flywheel(tmp_path):
    (tmp_path / "domain_router_map_flywheel_Qwen3-6-35B-A3B-UD-IQ2_M.json").write_text("{}")
    (tmp_path / "domain_router_map_Qwen3-6-35B-A3B-UD-IQ2_M.json").write_text("{}")
    found = find_router_table("Qwen3.6-35B-A3B-UD-IQ2_M.gguf", tmp_path)
    assert found.name == "domain_router_map_flywheel_Qwen3-6-35B-A3B-UD-IQ2_M.json"


def test_find_router_table_falls_back_to_exact_static(tmp_path):
    (tmp_path / "domain_router_map_ds.json").write_text("{}")
    found = find_router_table("ds.gguf", tmp_path)
    assert found.name == "domain_router_map_ds.json"


def test_find_router_table_returns_none(tmp_path):
    assert find_router_table("nope.gguf", tmp_path) is None


# ── top_k_for_coverage ─────────────────────────────────────────────

def test_top_k_for_coverage_exact_and_round_up():
    assert top_k_for_coverage(0.90) == 75
    assert top_k_for_coverage(0.93) == 100  # rounds up to next tier
    assert top_k_for_coverage(1.5) == 100   # above table → largest
    assert top_k_for_coverage(0.80) == 30


# ── compute_budget ─────────────────────────────────────────────────

def test_compute_budget_explicit_wins(monkeypatch):
    monkeypatch.setenv("MOE_L2_VRAM_BUDGET", "3.5")
    assert compute_budget(vram_budget_gb=6.0) == 6.0


def test_compute_budget_env(monkeypatch):
    monkeypatch.setenv("MOE_L2_VRAM_BUDGET", "3.5")
    assert compute_budget() == 3.5


def test_compute_budget_env_invalid_falls_through(monkeypatch):
    monkeypatch.setenv("MOE_L2_VRAM_BUDGET", "abc")
    assert compute_budget(vram_total_gb=10.0) == pytest.approx(10.0 * 0.60)


def test_compute_budget_detects_vram(monkeypatch):
    monkeypatch.delenv("MOE_L2_VRAM_BUDGET", raising=False)
    budget = compute_budget(vram_total_gb=8.0)
    assert budget == pytest.approx(4.8)


# ── filter_hot_domains ─────────────────────────────────────────────

def _old_style_table(domains=None, dom_freq=None):
    return {
        "model": "qwen",
        "dom_freq": dom_freq or {"qa": 50, "codegen": 30, "chat": 10},
        "domains": domains or {
            "qa": {"per_layer_domain_preferred": {"0": list(range(100))}},
            "codegen": {"per_layer_domain_preferred": {"0": [1, 2, 3]}},
            "chat": {"per_layer_domain_preferred": {"0": [7]}},
        },
    }


def test_filter_hot_domains_passthrough_new_format(tmp_path):
    p = tmp_path / "new.json"
    p.write_text(json.dumps({"layers": {"0": [1, 2]}}))
    assert filter_hot_domains(p) == p


def test_filter_hot_domains_uses_dom_freq_ranking(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps(_old_style_table()))
    out = filter_hot_domains(p, coverage_target=0.90)
    d = json.load(open(out))
    # hot ranking: qa(50) first → its layer-0 experts (100) truncated to top-75
    assert d["top_k"] == 75
    assert len(d["layers"]["0"]) == 75


def test_filter_hot_domains_n_hot_limit(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps(_old_style_table()))
    out = filter_hot_domains(p, n_hot=2, coverage_target=0.90)
    d = json.load(open(out))
    # qa + codegen only (top-2 by dom_freq); union = 100 experts → truncated to 75
    assert len(d["layers"]["0"]) == 75


# ── build_router_map_file ──────────────────────────────────────────

def test_build_router_map_file_with_existing_table(tmp_path, monkeypatch):
    table = tmp_path / "domain_router_map_flywheel_ds.json"
    table.write_text(json.dumps({
        "description": "flywheel",
        "model": "ds",
        "top_k": 5,
        "layers": {"0": [1, 2, 3, 4, 5], "1": [6, 7, 8]},
    }))
    out = build_router_map_file(
        "ds.gguf", router_top_k=10, data_dir=tmp_path, vram_budget_gb=50.0,
    )
    assert out and os.path.exists(out)
    with open(out) as f:
        lines = f.read().strip().splitlines()
    # [moe-l2 2026-08-14] 第一行是 EXPERT_TOTAL 元数据头，之后才是层数据
    assert lines[0].startswith("# EXPERT_TOTAL ")
    assert lines[1].startswith("0 ")
    assert lines[2].startswith("1 ")
    # top-k cap respected: all experts (≤5) kept under router_top_k=10
    assert len(lines[1].split()) == 6


def test_build_router_map_file_no_table_skips_collect_by_default(tmp_path, capsys):
    """[2026-09-19 决策] 冷启动采集默认关闭：无表 → 直接 on-demand，不跑 llama-cli。"""
    out = build_router_map_file("missing.gguf", data_dir=tmp_path, vram_budget_gb=50.0)
    assert out is None
    captured = capsys.readouterr().out
    assert "跳过冷启动采集" in captured
    assert "无路由表且无 llama-cli" not in captured  # 不再进可执行文件探测那条老路径


def test_build_router_map_file_auto_collect_optin_still_falls_back(tmp_path, capsys):
    """allow_auto_collect=True（CLI --auto-collect）仍走老路径；无可用 llama-cli 时降级提示。"""
    out = build_router_map_file("missing.gguf", data_dir=tmp_path, vram_budget_gb=50.0,
                                allow_auto_collect=True, llama_cli="")
    assert out is None
    assert "无路由表且无 llama-cli" in capsys.readouterr().out


def test_build_router_map_file_tiny_budget_on_demand(tmp_path):
    table = tmp_path / "domain_router_map_flywheel_ds.json"
    table.write_text(json.dumps({
        "layers": {"0": [1, 2, 3], "1": [4, 5, 6]},
    }))
    # budget below min tier (30 experts × 2 layers × 3 / 1000 × 0.82 ≈ 0.15GB)
    out = build_router_map_file("ds.gguf", data_dir=tmp_path, vram_budget_gb=0.01)
    assert out is None


# ── [走法 D 2026-09-19] 占位路由表 ─────────────────────────────────

def test_write_placeholder_router_map_header_only(tmp_path):
    from moe_l2.router_table import write_placeholder_router_map

    out = write_placeholder_router_map(4000)
    try:
        assert out.endswith(".router.map")
        with open(out, encoding="utf-8") as f:
            content = f.read()
        # 只有元数据头、没有层数据：引擎会打开 selective-pin 开关但不 pin 任何专家
        assert content == "# EXPERT_TOTAL 4000\n"
        assert len(content.strip().splitlines()) == 1
    finally:
        os.unlink(out)


def test_write_placeholder_router_map_clamps_to_min_one():
    from moe_l2.router_table import write_placeholder_router_map

    out = write_placeholder_router_map(0)
    try:
        with open(out, encoding="utf-8") as f:
            assert f.read() == "# EXPERT_TOTAL 1\n"
    finally:
        os.unlink(out)


def test_placeholder_router_map_for_uses_vram_slots(monkeypatch, capsys):
    from moe_l2 import vram_adaptive
    from moe_l2.router_table import placeholder_router_map_for

    monkeypatch.setattr(vram_adaptive, "auto_cache_slots", lambda *a, **kw: 18045)
    out = placeholder_router_map_for("m.gguf", 32768)
    assert out is not None
    try:
        with open(out, encoding="utf-8") as f:
            # 槽数 ÷ 3 张量（gate/up/down）→ 表里写专家数
            assert f.read() == f"# EXPERT_TOTAL {18045 // 3}\n"
        assert "占位路由表" in capsys.readouterr().out
    finally:
        os.unlink(out)


def test_placeholder_router_map_for_none_when_budget_unknown(monkeypatch, capsys):
    from moe_l2 import vram_adaptive
    from moe_l2.router_table import placeholder_router_map_for

    monkeypatch.setattr(vram_adaptive, "auto_cache_slots", lambda *a, **kw: None)
    assert placeholder_router_map_for("m.gguf", 32768) is None
    assert "占位表跳过" in capsys.readouterr().out


def test_placeholder_router_map_for_none_when_probe_raises(monkeypatch):
    from moe_l2 import vram_adaptive
    from moe_l2.router_table import placeholder_router_map_for

    def _boom(*a, **kw):
        raise RuntimeError("no nvidia-smi")

    monkeypatch.setattr(vram_adaptive, "auto_cache_slots", _boom)
    # 探测失败只降级、不抛异常（启动不能被它拦住）
    assert placeholder_router_map_for("m.gguf", 32768) is None
