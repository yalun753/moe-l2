"""moe_l2/vram_adaptive.py — 启动前显存探测 + 自动降档（context/parallel）

目的：llama-server 的 KV cache 按 parallel × n_ctx 预分配，显存不足时直接
cudaMalloc 失败崩溃（Qwen cache + parallel 4 在 11GB 卡 OOM 实测 2026-08-15）。
本模块在 cli 启动 server 前探测显存、按模型 KV 估算自动算 safe 参数：
  - 未指定 --ctx-size → 自动档 = 模型原生上限，按显存预算收敛（用户不用懂参数）
  - 显式指定 --ctx-size → 作为"期望值"，放不下仍自动降档（--ctx-force 跳过）
与 router_table 的"显存预算 → top-K 收敛"是同一套预算思想，这里管 KV 维度。

预算模型（2026-09-12 按实测重校）：
  固定开销 = 模型非专家权重（GGUF 张量表实测）+ BUFFER_OVERHEAD_GB 缓冲
  cache 预留 = 可用显存 × CACHE_RESERVE_RATIO（A3 专家 cache 懒分配 + prefill 峰值）
  KV 预算 = 可用显存 - 固定开销 - cache 预留
  n_ctx = min(想要, KV 预算 / 每 token KV)，放不下时按 CTX_LEVELS 向下收敛

KV 估算（2026-09-12 修正。旧公式对 qwen35moe 高估 16 倍，12G 卡请求 262144
被降档序列压回 8192，"256K 跑不动"实为估算错误而非显存不够）：
  kv_bytes_per_token = n_kv_layers × n_head_kv × (key_length + value_length) × 2 (F16)
  两个结构性修正：
    1) n_kv_layers = 真持 KV 的层数（含 blk.N.attn_k/attn_v 张量的层数）。
       混合线性注意力模型（qwen35moe：full_attention_interval=4）只有 40/4=10
       层是真注意力，旧公式按 block_count=40 多算 4 倍。
    2) head 维度优先 key_length/value_length 元数据（qwen35moe=256），
       退回 embedding/head_count（=128，少算 2 倍）。
  实测锚点（210 / RTX 3060 12G / Qwen3.6-35B-A3B UD-IQ2_M，2026-09-09 ctx 扫描）：
    斜率 20.3-20.8 KB/tok；本公式 20.0 KB/tok，乘 KV_SAFETY 后 23.0 KB/tok。
"""

from __future__ import annotations

import functools
import subprocess

# KV 之外给 A3 专家 cache / prefill 峰值留的份额（占可用显存）
CACHE_RESERVE_RATIO = 0.25
# compute/graph/output/host 缓冲（210 实测：CUDA0 compute 283 MiB + CUDA_Host 40 MiB
# + output 1 MiB ≈ 0.32GB，留 0.45GB 余量）
BUFFER_OVERHEAD_GB = 0.45
# 读不到模型张量表时的固定开销兜底（GB）
FIXED_VRAM_GB_FALLBACK = 2.0
# KV 估算安全系数（公式 20.0 KB/tok vs 实测 20.3-20.8 KB/tok，留 ~15% 余量）
KV_SAFETY = 1.15
# context 降档档位（从高到低）。旧表缺 16384/32768，放不下 32768 会直接跳回 8192
CTX_LEVELS = [1048576, 524288, 262144, 131072, 65536, 32768, 16384,
              8192, 4096, 2048, 1024, 512, 256]
# 自动档上限：更高档（>256K）无实测数据，不自动给，需显式 --ctx-size
AUTO_CTX_MAX = 262144
# 自动档兜底（模型信息缺失/预算算不出时退回历史默认，避免放行原生上限导致 OOM）
AUTO_CTX_FALLBACK = 8192
# 显存预算占总显存比例（与 router_table.DEFAULT_VRAM_BUDGET_RATIO 一致）
VRAM_BUDGET_RATIO = 0.6
# MLA 架构（KV 极小，不按 GQA 公式降档）
MLA_ARCHS = ("deepseek2", "dsv4", "deepseek-v2", "deepseek-v3", "deepseek-v4")

# GGUF metadata 字段键（与 gguf_reader 的 _LAYER_KEYS 同风格）
_LAYER_KEYS = ["qwen2moe.block_count", "deepseek2.block_count", "llama.block_count",
               "qwen2.block_count", "deepseek3.block_count"]
_KV_HEAD_KEYS = ["qwen2moe.attention.head_count_kv", "deepseek2.attention.head_count_kv",
                 "llama.attention.head_count_kv", "qwen2.attention.head_count_kv",
                 "deepseek3.attention.head_count_kv"]
_HEAD_KEYS = ["qwen2moe.attention.head_count", "deepseek2.attention.head_count",
              "llama.attention.head_count", "qwen2.attention.head_count",
              "deepseek3.attention.head_count"]
_HEAD_DIM_KEYS = ["qwen2moe.attention.head_dim", "deepseek2.attention.head_dim",
                  "llama.attention.head_dim", "qwen2.attention.head_dim",
                  "deepseek3.attention.head_dim"]
_EMBED_KEYS = ["qwen2moe.embedding_length", "deepseek2.embedding_length",
               "llama.embedding_length", "qwen2.embedding_length",
               "deepseek3.embedding_length"]
_ARCH_KEYS = ["general.architecture"]

# MoE 专家权重张量名特征（非专家权重 = 其余全部，跟着 -ngl 99 上 GPU）
_EXPERT_NAME_MARK = "_exps"


def probe_vram_mb() -> tuple[int, int]:
    """探测 GPU 总显存/可用显存（MB）。nvidia-smi 失败返回 (0, 0)。"""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        line = out.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


@functools.lru_cache(maxsize=8)
def _model_facts(model_path: str) -> dict:
    """一次扫 GGUF 取全部事实（metadata + 张量表），带缓存。

    返回：
      meta: dict              归一化后的关键元数据
      context_length: int     模型原生上下文上限（None = 读不到）
      n_kv_layers: int        真持 KV 的层数（混合线性注意力模型 < block_count）
      n_layers: int           block_count（张量表实测）
      non_expert_bytes: int   非专家权重字节数（→ 固定显存开销）
    """
    facts: dict = {"meta": {}, "context_length": None, "n_kv_layers": None,
                   "n_layers": None, "non_expert_bytes": 0}
    try:
        from gguf import GGUFReader  # 与 gguf_reader.py 同一依赖
        reader = GGUFReader(model_path)

        def _get(key: str):
            field = reader.get_field(key)
            if field is None:
                return None
            raw = field.parts[-1]
            try:
                if field.types[-1] == 8:  # STRING
                    return bytes(raw).decode("utf-8", errors="replace")
                return int(raw.item())
            except Exception:
                return None

        meta: dict = {}
        arch = _get("general.architecture")
        if arch:
            meta["general.architecture"] = arch
            # 动态拼字段：<arch>.<field>，同时归一化到通用键，下游按通用键取值
            norm_map = {"block_count": "block_count",
                        "context_length": "context_length",
                        "attention.head_count_kv": "head_count_kv",
                        "attention.head_count": "head_count",
                        "attention.head_dim": "head_dim",
                        "attention.key_length": "key_length",
                        "attention.value_length": "value_length",
                        "full_attention_interval": "full_attention_interval",
                        "embedding_length": "embedding_length"}
            for field, norm in norm_map.items():
                v = _get(f"{arch}.{field}")
                if v is not None:
                    meta[f"{arch}.{field}"] = v
                    meta[norm] = v
        # 兜底：老架构名
        if not any(k.endswith("block_count") for k in meta):
            for key in (_LAYER_KEYS + _KV_HEAD_KEYS + _HEAD_KEYS
                        + _HEAD_DIM_KEYS + _EMBED_KEYS):
                v = _get(key)
                if v is not None:
                    meta[key] = v

        # 张量表扫一遍：KV 层数 + 非专家权重体积（只读头部，不读权重数据）
        kv_layers: set[int] = set()
        n_layers = 0
        non_expert = 0
        for t in reader.tensors:
            name = t.name
            if name.startswith("blk."):
                try:
                    li = int(name.split(".", 2)[1])
                    n_layers = max(n_layers, li + 1)
                except (IndexError, ValueError):
                    li = -1
                if name.endswith(("attn_k.weight", "attn_v.weight")) and li >= 0:
                    kv_layers.add(li)
            if _EXPERT_NAME_MARK not in name:
                non_expert += int(getattr(t, "n_bytes", 0) or 0)

        facts["meta"] = meta
        facts["n_layers"] = n_layers or meta.get("block_count") or None
        facts["non_expert_bytes"] = non_expert
        ctx_len = meta.get("context_length")
        facts["context_length"] = int(ctx_len) if ctx_len else None
        if kv_layers:
            facts["n_kv_layers"] = len(kv_layers)
        elif facts["n_layers"]:
            # 没有 attn_k/attn_v 张量表信息时，用 full_attention_interval 推
            interval = meta.get("full_attention_interval") or 1
            interval = int(interval) if int(interval) > 0 else 1
            facts["n_kv_layers"] = max(1, facts["n_layers"] // interval)
        reader.close()
    except Exception:
        pass
    return facts


def _read_gguf_meta(model_path: str) -> dict:
    """读 GGUF 关键 metadata（归一化字段）。失败返回空 dict。"""
    return _model_facts(model_path).get("meta", {})


def _arch_is_mla(meta: dict) -> bool:
    for k in _ARCH_KEYS:
        v = meta.get(k)
        if v and any(a in str(v).lower() for a in MLA_ARCHS):
            return True
    return False


def estimate_kv_bytes_per_token(model_path: str) -> float:
    """估算每 token 的 KV cache 字节数（F16，K+V）。

    kv_bytes = n_kv_layers × head_count_kv × (key_length + value_length) × 2 (F16)
    再乘 KV_SAFETY。MLA 架构（deepseek2 系列）KV 是 latent 压缩（32K 上下文仅
    几十 MiB），直接返回小值（按 1MB/32K token 估）避免公式高估导致误降档。
    """
    facts = _model_facts(model_path)
    meta = facts["meta"]
    if not meta:
        return 0.0  # 未知模型 → 不降档（保守放行，崩了由用户手动降）

    if _arch_is_mla(meta):
        # MLA：~32K 上下文几十 MiB → 按 1MB/32K token 估
        return 1024 * 1024 / 32768

    def _first(keys):
        # 通用键优先（_model_facts 归一化），再兜底固定列表
        for k in keys:
            if k in meta:
                return meta[k]
        return None

    n_kv_layers = facts.get("n_kv_layers") or _first(["block_count"] + _LAYER_KEYS)
    n_kv_heads = _first(["head_count_kv"] + _KV_HEAD_KEYS)
    n_heads = _first(["head_count"] + _HEAD_KEYS)
    embed = _first(["embedding_length"] + _EMBED_KEYS)

    if not n_kv_layers or not n_kv_heads:
        return 0.0
    head_dim_k = _first(["key_length", "head_dim"] + _HEAD_DIM_KEYS)
    head_dim_v = _first(["value_length", "head_dim"] + _HEAD_DIM_KEYS)
    if not head_dim_k or not head_dim_v:
        if n_heads and embed:
            head_dim_k = head_dim_k or embed // n_heads
            head_dim_v = head_dim_v or embed // n_heads
        else:
            return 0.0

    kv_per_token = n_kv_layers * n_kv_heads * (head_dim_k + head_dim_v) * 2  # K+V, F16
    return float(kv_per_token) * KV_SAFETY


def estimate_fixed_vram_gb(model_path: str) -> float:
    """固定显存开销（GB）= 模型非专家权重 + 缓冲。读不到 → 兜底常量。

    非专家权重跟着 -ngl 99 全部上 GPU（210 实测 CUDA0 model buffer 1330.99 MiB
    for Qwen3.6-35B-A3B UD-IQ2_M），不同量化档差别很大（Q4 的非专家层明显更大），
    所以按模型实测而不是拍一个常量。
    """
    non_expert = _model_facts(model_path).get("non_expert_bytes") or 0
    if non_expert <= 0:
        return FIXED_VRAM_GB_FALLBACK
    return non_expert / 1024 ** 3 + BUFFER_OVERHEAD_GB


def compute_safe_params(
    model_path: str,
    vram_total_mb: int | None = None,
    vram_free_mb: int | None = None,
    want_ctx: int | None = None,
    want_parallel: int = 1,
    force_ctx: bool = False,
) -> dict:
    """根据显存算 safe (n_ctx, n_parallel, kv_budget_gb, reason, auto)。

    返回 dict：
      n_ctx: int          最终 context
      n_parallel: int     并行度（先保 1；富余时给 want_parallel）
      kv_budget_gb: float KV 可用预算
      reason: str         说明（自动档 / "OK" / 降档原因）
      auto: bool          True = 未显式指定 ctx（走自动档）

    want_ctx=None → 自动档：以模型原生上限（上限 AUTO_CTX_MAX）为目标，按显存预算收敛。
    want_ctx=整数 → 期望值，放不下仍自动降档；force_ctx=True 时完全信任该值。
    """
    facts = _model_facts(model_path)
    native_ctx = facts.get("context_length")
    auto = want_ctx is None
    if auto:
        want_ctx = int(min(native_ctx or AUTO_CTX_MAX, AUTO_CTX_MAX))
    n_parallel = want_parallel if want_parallel > 1 else 1

    if force_ctx:
        reason = (f"OK (ctx-force: 未指定 --ctx-size，直接用自动档 {int(want_ctx)} 并跳过降档)"
                  if auto else
                  f"OK (ctx-force: 显式指定 {int(want_ctx)}，跳过自动降档)")
        return {"n_ctx": int(want_ctx), "n_parallel": n_parallel,
                "kv_budget_gb": 0.0, "auto": auto, "reason": reason}

    total_mb, free_mb = probe_vram_mb()
    if vram_total_mb is not None:
        total_mb = vram_total_mb
    if vram_free_mb is not None:
        free_mb = vram_free_mb

    # 无显存信息（非 N 卡/驱动问题）→ 不降档
    if total_mb <= 0 or free_mb <= 0:
        return {"n_ctx": int(want_ctx), "n_parallel": n_parallel,
                "kv_budget_gb": 0.0, "auto": auto,
                "reason": "OK (no vram info)"}

    free_gb = free_mb / 1024.0
    fixed_gb = estimate_fixed_vram_gb(model_path)
    kv_budget_gb = free_gb - fixed_gb - free_gb * CACHE_RESERVE_RATIO
    kv_per_token = estimate_kv_bytes_per_token(model_path)

    # 预算算不出（负预算/模型信息缺失）：
    #   自动档 → 退回历史默认 8192，绝不放行原生上限（否则必 OOM）；
    #   显式值 → 保守放行（用户自己知道显存放得下，2026-08-15 起的既定行为）
    if kv_budget_gb <= 0 or kv_per_token <= 0:
        if auto:
            n_ctx = int(min(want_ctx, AUTO_CTX_FALLBACK))
            return {"n_ctx": n_ctx, "n_parallel": n_parallel,
                    "kv_budget_gb": round(max(kv_budget_gb, 0.0), 2), "auto": True,
                    "reason": (f"ctx 自动档 {n_ctx}（模型信息缺失或显存预算不足，"
                               f"退回默认；如确定放得下用 --ctx-size N --ctx-force）")}
        return {"n_ctx": int(want_ctx), "n_parallel": n_parallel,
                "kv_budget_gb": round(max(kv_budget_gb, 0.0), 2), "auto": False,
                "reason": "OK (预算不足，放行显式值)"}

    max_tokens = int(kv_budget_gb * 1024 ** 3 / kv_per_token)
    n_ctx = int(want_ctx)
    reason = "OK"
    if auto:
        reason = (f"ctx 自动档 {n_ctx}（模型原生上限 {native_ctx or '未知'}，"
                  f"KV 预算 {kv_budget_gb:.1f}GB ≈ {max_tokens} tok）")

    # 降档顺序：先降 parallel（Qwen OOM 主因是 parallel 4 的 KV 翻倍），
    # 再降 ctx。want_ctx × want_parallel 能 fit 则保持。
    need = n_ctx * n_parallel
    if need > max_tokens:
        # 尝试保持 want_ctx，降 parallel
        p = 1
        while p <= want_parallel and n_ctx * p <= max_tokens:
            p += 1
        p -= 1
        if p >= 1 and n_ctx * p <= max_tokens:
            n_parallel = p
            if p < want_parallel:
                reason = (f"OOM 风险：KV 预算 {kv_budget_gb:.1f}GB（~{max_tokens} tok）"
                          f"→ parallel 降档 {want_parallel}→{p}（ctx 保持 {n_ctx}）")
        else:
            # parallel 降到 1 仍不够 → 降 ctx（按 CTX_LEVELS 取最大可放下的档）
            n_parallel = 1
            chosen = None
            for level in CTX_LEVELS:
                if level <= max_tokens:
                    chosen = level
                    break
            if chosen is None:
                chosen = CTX_LEVELS[-1]
            n_ctx = int(min(chosen, want_ctx))
            if auto:
                reason = (f"ctx 自动档 {n_ctx}（模型原生上限 {native_ctx or '未知'} 超出"
                          f"显存预算 {kv_budget_gb:.1f}GB ≈ {max_tokens} tok，已收敛）")
            else:
                reason = (f"OOM 风险：KV 预算 {kv_budget_gb:.1f}GB（~{max_tokens} tok）"
                          f"→ 降档 parallel {want_parallel}→1 + context {want_ctx}→{n_ctx}")

    return {"n_ctx": n_ctx, "n_parallel": n_parallel,
            "kv_budget_gb": round(kv_budget_gb, 2), "reason": reason, "auto": auto}


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else ""
    if model:
        f = _model_facts(model)
        print(f"context_length : {f.get('context_length')}")
        print(f"n_layers/kv    : {f.get('n_layers')} / {f.get('n_kv_layers')}")
        print(f"non-expert GB  : {f.get('non_expert_bytes', 0) / 1024 ** 3:.2f}")
        print(f"fixed GB       : {estimate_fixed_vram_gb(model):.2f}")
        print(f"kv/ctx         : {estimate_kv_bytes_per_token(model):.0f} B/token")
        print(compute_safe_params(model))
    else:
        print("usage: python3 -m moe_l2.vram_adaptive <model.gguf>")
