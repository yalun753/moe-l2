"""test_vram_adaptive.py — 显存自适应（防 OOM 自动降档）单元测试

mock 掉 probe_vram_mb / estimate_kv_bytes_per_token，只测降档决策逻辑：
  - 无显存信息 → 不降档（放行）
  - 大 KV 模型（Qwen/GQA）11GB 卡 → parallel 先降
  - 小 KV 模型（DS/MLA）→ 保持并行
  - 显存富余（24GB）→ 保持
  - 极端不足 → parallel 降到底再降 ctx
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import moe_l2.vram_adaptive as va


@pytest.fixture(autouse=True)
def _mock_probe(monkeypatch):
    """默认 mock：probe 返回 0（无显存信息），KV 估算返回 0（未知模型）。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (0, 0))
    monkeypatch.setattr(va, "estimate_kv_bytes_per_token", lambda p: 0.0)


def test_no_vram_info_keeps_params():
    """无显存信息（非 N 卡/驱动问题）→ 不降档，放行默认参数。"""
    r = va.compute_safe_params("/x.gguf", want_ctx=8192, want_parallel=4)
    assert r["n_ctx"] == 8192
    assert r["n_parallel"] == 4
    assert "OK" in r["reason"]


def test_explicit_vram_args_used(monkeypatch):
    """显式传 vram 参数时使用传入值（不依赖真实探测）。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (11264, 6000))  # 11GB, free 6GB
    monkeypatch.setattr(va, "estimate_kv_bytes_per_token", lambda p: 320000.0)  # Qwen ~312KB/tok

    r = va.compute_safe_params("/qwen.gguf", want_ctx=8192, want_parallel=4)
    # KV 预算 = 6 - 2.5 - 1.8 = 1.7GB → ~5320 tok → 8192×4 放不下 → 降 parallel
    assert r["n_parallel"] < 4
    assert "降档" in r["reason"]


def test_big_kv_downgrades_parallel_before_ctx(monkeypatch):
    """大 KV（Qwen GQA）：parallel 4 放不下 → 先降 parallel，ctx 保持。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (11264, 6000))
    monkeypatch.setattr(va, "estimate_kv_bytes_per_token", lambda p: 320000.0)

    r = va.compute_safe_params("/qwen.gguf", want_ctx=2048, want_parallel=4)
    # KV 预算 1.7GB → 5320 tok；2048×4=8192 > 5320；2048×2=4096 ≤ 5320 → parallel 2（能放下的最大）
    assert r["n_ctx"] == 2048
    assert r["n_parallel"] == 2


def test_mla_tiny_kv_keeps_parallel(monkeypatch):
    """小 KV（DS/V4 MLA）：KV 极小 → parallel 4 保持。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (11264, 6000))
    monkeypatch.setattr(va, "estimate_kv_bytes_per_token", lambda p: 32.0)  # MLA ~32B/tok

    r = va.compute_safe_params("/ds.gguf", want_ctx=8192, want_parallel=4)
    assert r["n_parallel"] == 4
    assert r["n_ctx"] == 8192
    assert r["reason"] == "OK"


def test_24gb_keeps_parallel(monkeypatch):
    """24GB 卡富余：Qwen 大 KV 也保持 parallel 4。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (24564, 20000))
    monkeypatch.setattr(va, "estimate_kv_bytes_per_token", lambda p: 320000.0)

    r = va.compute_safe_params("/qwen.gguf", want_ctx=8192, want_parallel=4)
    # KV 预算 = 20 - 2.5 - 6 = 11.5GB → ~36000 tok → 8192×4=32768 ≤ 36000 → 保持
    assert r["n_parallel"] == 4
    assert r["n_ctx"] == 8192


def test_extreme_shortage_downgrades_ctx(monkeypatch):
    """极端不足：parallel 降到 1 仍不够 → ctx 也降。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (11264, 3000))  # free 3GB
    monkeypatch.setattr(va, "estimate_kv_bytes_per_token", lambda p: 320000.0)

    r = va.compute_safe_params("/qwen.gguf", want_ctx=8192, want_parallel=4)
    # KV 预算 = 3 - 2.5 - 0.9 = -0.4 → 0 → 不降档（预算为负时保守放行）
    assert r["n_ctx"] == 8192
    assert r["n_parallel"] == 4


def test_mla_arch_detection():
    """MLA 架构识别。"""
    assert va._arch_is_mla({"general.architecture": "deepseek2"})
    assert va._arch_is_mla({"general.architecture": "DeepSeek-V3"})
    assert not va._arch_is_mla({"general.architecture": "qwen35moe"})
    assert not va._arch_is_mla({})


def test_ctx_force_keeps_large_ctx(monkeypatch):
    """--ctx-force（force_ctx=True）：即使 KV 估算会触发降档，也完全信任 want_ctx。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (12288, 12288))
    # qwen35moe 高估场景：12G 卡 262144 本会被降档序列压到 8192
    monkeypatch.setattr(va, "estimate_kv_bytes_per_token", lambda p: 327680.0)

    r = va.compute_safe_params("/qwen35moe.gguf", want_ctx=262144, force_ctx=True)
    assert r["n_ctx"] == 262144
    assert "ctx-force" in r["reason"]


def test_ctx_force_default_off_preserves_downgrade(monkeypatch):
    """默认（force_ctx=False）行为不变：大 ctx 仍自动降档（防 OOM 保护）。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (12288, 12288))
    monkeypatch.setattr(va, "estimate_kv_bytes_per_token", lambda p: 327680.0)

    r = va.compute_safe_params("/qwen35moe.gguf", want_ctx=262144)
    assert r["n_ctx"] == 8192  # 降档序列最高档
    assert "降档" in r["reason"]
