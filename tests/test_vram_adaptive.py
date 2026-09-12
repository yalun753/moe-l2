"""test_vram_adaptive.py — 显存自适应（防 OOM 自动降档 + ctx 自动档）单元测试

主要 mock 掉 probe_vram_mb / _model_facts（GGUF 读取），只测决策逻辑：
  - 无显存信息 → 不降档（放行）
  - 自动档（want_ctx=None）→ 取模型原生上限，按预算收敛
  - 降档顺序：先降 parallel，再降 ctx；档位表含 16384/32768（旧表缺失）
  - 混合线性注意力模型的 KV 估算（n_kv_layers < block_count、key_length 优先）
  - 预算算不出时：自动档退回 8192，显式值放行（既定行为）
  - --ctx-force 完全信任显式值
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import moe_l2.vram_adaptive as va

# 真实实现（autouse fixture 会把 va._model_facts 换成假的，这里留个真身给 GGUF 实测用例）
_REAL_MODEL_FACTS = va._model_facts


def _facts(ctx=262144, kv_layers=10, layers=40, non_expert=1_330_000_000, **meta):
    """qwen35moe 实况（210 实测）：40 层、每 4 层 1 层真注意力、head_dim 256。"""
    m = {"general.architecture": "qwen35moe", "block_count": layers,
         "head_count_kv": 2, "head_count": 16, "key_length": 256, "value_length": 256,
         "full_attention_interval": 4, "embedding_length": 2048}
    m.update(meta)
    return {"meta": m, "context_length": ctx, "n_kv_layers": kv_layers,
            "n_layers": layers, "non_expert_bytes": non_expert}


@pytest.fixture(autouse=True)
def _mock_probe(monkeypatch):
    """默认 mock：无显存信息（不降档）+ 标准 qwen35moe 事实。各用例按需覆盖。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (0, 0))
    monkeypatch.setattr(va, "_model_facts", lambda p: _facts())


def test_no_vram_info_keeps_params():
    """无显存信息（非 N 卡/驱动问题）→ 不降档，放行。"""
    r = va.compute_safe_params("/x.gguf", want_ctx=8192, want_parallel=4)
    assert r["n_ctx"] == 8192
    assert r["n_parallel"] == 4
    assert "OK" in r["reason"]
    assert r["auto"] is False


def test_auto_ctx_uses_native_context_length(monkeypatch):
    """自动档：未指定 ctx → 取模型原生上限，12G 卡不降档（旧版被压到 8192）。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (12288, 11255))

    r = va.compute_safe_params("/qwen.gguf")
    assert r["n_ctx"] == 262144
    assert r["auto"] is True
    assert "自动档" in r["reason"]


def test_auto_ctx_narrows_on_small_vram(monkeypatch):
    """自动档在显存不足时收敛到档位表内的值（不放行原生 262144）。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (8192, 4000))

    r = va.compute_safe_params("/qwen.gguf")
    assert r["n_ctx"] == 32768
    assert r["n_ctx"] in va.CTX_LEVELS
    assert r["auto"] is True
    assert "自动档" in r["reason"]


def test_ladder_has_16k_and_32k(monkeypatch):
    """档位表补了 16384/32768：放得下 16K 就不再直接跳回 8192（旧表缺陷）。"""
    assert 16384 in va.CTX_LEVELS and 32768 in va.CTX_LEVELS
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (8192, 2800))  # 预算 ≈ 16.5K tok

    r = va.compute_safe_params("/qwen.gguf", want_ctx=32768)
    assert r["n_ctx"] == 16384
    assert r["auto"] is False
    assert "降档" in r["reason"]


def test_explicit_ctx_downgrades_to_ladder(monkeypatch):
    """显式 ctx 放不下 → 按档位表收敛（不再跳到 8192）。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (8192, 2800))

    r = va.compute_safe_params("/qwen.gguf", want_ctx=131072)
    assert r["n_ctx"] == 16384


def test_parallel_downgraded_before_ctx(monkeypatch):
    """大 KV：parallel 4 放不下 → 先降 parallel，ctx 保持。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (11264, 6000))  # 预算 ≈ 120K tok

    r = va.compute_safe_params("/qwen.gguf", want_ctx=32768, want_parallel=4)
    assert r["n_ctx"] == 32768
    assert 1 <= r["n_parallel"] < 4
    assert "parallel 降档" in r["reason"]


def test_mla_tiny_kv_keeps_parallel(monkeypatch):
    """小 KV（DS/V4 MLA）：KV 极小 → parallel 4 保持。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (11264, 6000))
    monkeypatch.setattr(va, "estimate_kv_bytes_per_token", lambda p: 32.0)

    r = va.compute_safe_params("/ds.gguf", want_ctx=8192, want_parallel=4)
    assert r["n_parallel"] == 4
    assert r["n_ctx"] == 8192
    assert r["reason"] == "OK"


def test_24gb_keeps_parallel(monkeypatch):
    """24GB 卡富余：parallel 4 保持。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (24564, 20000))

    r = va.compute_safe_params("/qwen.gguf", want_ctx=8192, want_parallel=4)
    assert r["n_parallel"] == 4
    assert r["n_ctx"] == 8192


def test_budget_uncomputable_explicit_passes_auto_falls_back(monkeypatch):
    """预算为负：显式值放行（既定行为），自动档退回 8192（不放行原生上限）。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (8192, 2000))

    r = va.compute_safe_params("/qwen.gguf", want_ctx=8192, want_parallel=4)
    assert r["n_ctx"] == 8192
    assert r["n_parallel"] == 4

    r2 = va.compute_safe_params("/qwen.gguf")
    assert r2["n_ctx"] == va.AUTO_CTX_FALLBACK
    assert r2["auto"] is True


def test_mla_arch_detection():
    """MLA 架构识别。"""
    assert va._arch_is_mla({"general.architecture": "deepseek2"})
    assert va._arch_is_mla({"general.architecture": "DeepSeek-V3"})
    assert not va._arch_is_mla({"general.architecture": "qwen35moe"})
    assert not va._arch_is_mla({})


def test_kv_estimate_hybrid_attention():
    """KV 估算按真持 KV 的层数（40 层里只有 10 层真注意力）+ key_length 优先。

    210 实测斜率 20.3-20.8 KB/tok（Qwen3.6-35B-A3B UD-IQ2_M / RTX 3060 12G）：
    10 × 2 × (256 + 256) × 2 = 20480 B/tok，乘安全系数 1.15 = 23552。
    """
    got = va.estimate_kv_bytes_per_token("/qwen.gguf")
    assert got == pytest.approx(10 * 2 * (256 + 256) * 2 * va.KV_SAFETY)
    assert 20_000 < got < 24_000  # 旧公式 327680（高估 16 倍）


def test_kv_estimate_falls_back_to_embed_over_heads(monkeypatch):
    """没有 key_length/value_length 时退回 embedding/head_count，仍按 KV 层数算。"""
    monkeypatch.setattr(
        va, "_model_facts",
        lambda p: _facts(key_length=None, value_length=None),
    )
    # key_length/value_length 被显式设 None → meta 里取不到 → 128 = 2048/16
    facts = _facts()
    facts["meta"].pop("key_length")
    facts["meta"].pop("value_length")
    monkeypatch.setattr(va, "_model_facts", lambda p: facts)

    got = va.estimate_kv_bytes_per_token("/qwen.gguf")
    assert got == pytest.approx(10 * 2 * (128 + 128) * 2 * va.KV_SAFETY)


def test_kv_estimate_unknown_model_is_zero(monkeypatch):
    """读不到元数据 → 0（不降档，保守放行）。"""
    monkeypatch.setattr(va, "_model_facts", lambda p: {"meta": {}})
    assert va.estimate_kv_bytes_per_token("/unknown.gguf") == 0.0


def test_model_facts_from_real_gguf(tmp_path):
    """真写一个 GGUF 验证 _model_facts 的实测口径（混合注意力层数 / 非专家体积）。

    这是 KV 估算与固定开销的输入端：数错层 → 又回到 16 倍高估。
    """
    gguf = pytest.importorskip("gguf")
    import numpy as np

    path = tmp_path / "mini-qwen35moe.gguf"
    w = gguf.GGUFWriter(str(path), "qwen35moe")
    w.add_context_length(262144)
    w.add_block_count(8)
    w.add_embedding_length(2048)
    w.add_head_count(16)
    w.add_head_count_kv(2)
    w.add_key_length(256)
    w.add_value_length(256)
    w.add_uint32("qwen35moe.full_attention_interval", 4)
    # 8 层里只有 blk.3 是真注意力（attn_k/attn_v），其余是 SSM/线性层
    w.add_tensor("blk.0.ssm_conv1d.weight", np.zeros((4, 2048), dtype=np.float32))
    w.add_tensor("blk.1.ffn_gate_exps.weight", np.zeros((8, 512, 2048), dtype=np.float32))
    w.add_tensor("blk.7.ffn_down_exps.weight", np.zeros((8, 2048, 512), dtype=np.float32))
    w.add_tensor("blk.3.attn_k.weight", np.zeros((256, 2048), dtype=np.float32))
    w.add_tensor("blk.3.attn_v.weight", np.zeros((256, 2048), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    facts = _REAL_MODEL_FACTS(str(path))
    assert facts["context_length"] == 262144
    assert facts["n_layers"] == 8
    assert facts["n_kv_layers"] == 1  # 只有 blk.3 持 KV（不是 block_count=8）
    # 非专家 = 张量总字节 - 带 _exps 的专家张量
    tensor_bytes = (4 * 2048 + 8 * 512 * 2048 + 8 * 2048 * 512 + 256 * 2048 + 256 * 2048) * 4
    expert_bytes = (8 * 512 * 2048 + 8 * 2048 * 512) * 4
    assert facts["non_expert_bytes"] == tensor_bytes - expert_bytes
    assert facts["meta"]["key_length"] == 256


def test_fixed_vram_from_model_tensors(monkeypatch):
    """固定开销按模型非专家权重实测（Q4 的非专家层比 IQ2 大，不能拍常量）。"""
    monkeypatch.setattr(va, "_model_facts", lambda p: _facts(non_expert=3_000_000_000))
    assert va.estimate_fixed_vram_gb("/qwen.gguf") == pytest.approx(
        3_000_000_000 / 1024 ** 3 + va.BUFFER_OVERHEAD_GB
    )
    # 读不到张量表 → 兜底常量
    monkeypatch.setattr(va, "_model_facts", lambda p: {"meta": {}, "non_expert_bytes": 0})
    assert va.estimate_fixed_vram_gb("/x.gguf") == va.FIXED_VRAM_GB_FALLBACK


def test_ctx_force_keeps_large_ctx(monkeypatch):
    """--ctx-force + 显式 ctx：即使估算会降档，也完全信任该值。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (12288, 12288))

    r = va.compute_safe_params("/qwen.gguf", want_ctx=262144, force_ctx=True)
    assert r["n_ctx"] == 262144
    assert "ctx-force" in r["reason"]
    assert r["auto"] is False


def test_ctx_force_without_explicit_ctx_keeps_auto(monkeypatch):
    """--ctx-force 单独出现（无 --ctx-size）：用自动档并跳过降档。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (12288, 11255))

    r = va.compute_safe_params("/qwen.gguf", force_ctx=True)
    assert r["n_ctx"] == 262144
    assert r["auto"] is True
    assert "ctx-force" in r["reason"]


def test_ctx_force_default_off_preserves_downgrade(monkeypatch):
    """默认（force_ctx=False）行为：显式大 ctx 仍自动降档（防 OOM 保护）。"""
    monkeypatch.setattr(va, "probe_vram_mb", lambda: (12288, 11255))

    r = va.compute_safe_params("/qwen.gguf", want_ctx=1048576)
    assert r["n_ctx"] == 262144  # 12G + IQ2 的实测甜点档（旧版被压到 8192）
    assert "降档" in r["reason"]
