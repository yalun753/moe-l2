"""moe_l2/vram_adaptive.py — 启动前显存探测 + 自动降档（context/parallel）

目的：llama-server 的 KV cache 按 parallel × n_ctx 预分配，显存不足时直接
cudaMalloc 失败崩溃（Qwen cache + parallel 4 在 11GB 卡 OOM 实测 2026-08-15）。
本模块在 cli 启动 server 前探测显存、按模型 KV 估算自动算 safe 参数：
  - 显存富余 → 保持用户要的 context/parallel
  - 显存不足 → 自动降 context（8192→4096→2048→1024→512），保 parallel=1
与 router_table 的"显存预算 → top-K 收敛"是同一套预算思想，这里管 KV 维度。

设计（2026-08-15）：
  预算 = 可用显存 × 0.6（与 compute_budget 同比例）
  固定开销 ≈ 2.5GB（非专家层 ~1.3GB + graph/计算缓冲 ~1.2GB）
  专家 cache 预算 = 预算 × 0.6（router_table top-k 收敛用）
  KV 预算 = 预算 - 固定开销 - cache 预算（保底 > 0）
  max_ctx_tokens = KV 预算 / kv_bytes_per_token
  n_ctx = min(想要, max_ctx_tokens) 向下取整到 512 倍数
  n_parallel = 1（先保单路；富余时可选升）
"""

from __future__ import annotations

import subprocess

# 固定显存开销（非专家层 + graph/计算缓冲），GB
FIXED_VRAM_GB = 2.5
# 预算中分给专家 cache 的比例（其余给 KV）
CACHE_BUDGET_RATIO = 0.6
# 显存预算占总显存比例（与 router_table.DEFAULT_VRAM_BUDGET_RATIO 一致）
VRAM_BUDGET_RATIO = 0.6
# context 降档序列
CTX_LEVELS = [8192, 4096, 2048, 1024, 512, 256]
# KV 估算安全系数（公式低估实测 ~11 倍：Qwen c2048 一份 KV 1.9GB ≈ 928KB/tok，
# 公式 80KB/tok；乘 8 保守，宁降档不 OOM）
KV_SAFETY = 8.0
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


def _read_gguf_meta(model_path: str) -> dict:
    """读 GGUF 关键 metadata（架构/层数/KV 头/head_dim/embedding）。失败返回空 dict。

    字段键 = "<architecture>.<field>" 动态拼接（如 qwen35moe.block_count），
    兼容任意架构（qwen2moe/qwen35moe/deepseek2/llama...）。
    """
    meta: dict = {}
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

        arch = _get("general.architecture")
        if arch:
            meta["general.architecture"] = arch
            # 动态拼字段：<arch>.block_count / .attention.head_count_kv 等
            # 同时归一化到通用键（block_count / head_count_kv / ...），
            # 下游 estimate 用通用键即可，兼容任意架构。
            norm_map = {"block_count": "block_count",
                        "attention.head_count_kv": "head_count_kv",
                        "attention.head_count": "head_count",
                        "attention.head_dim": "head_dim",
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
        reader.close()
    except Exception:
        pass
    return meta


def _arch_is_mla(meta: dict) -> bool:
    for k in _ARCH_KEYS:
        v = meta.get(k)
        if v and any(a in str(v).lower() for a in MLA_ARCHS):
            return True
    return False


def estimate_kv_bytes_per_token(model_path: str) -> float:
    """估算每 token 的 KV cache 字节数（F16，K+V）。

    kv_bytes = 2 (K+V) × block_count × head_count_kv × head_dim × 2 (F16 bytes)
    MLA 架构（deepseek2 系列）KV 是 latent 压缩（32K 上下文仅几十 MiB），
    直接返回小值（按 1MB/32K token 估）避免公式高估导致误降档。
    """
    meta = _read_gguf_meta(model_path)
    if not meta:
        return 0.0  # 未知模型 → 不降档（保守放行，崩了由用户手动降）

    if _arch_is_mla(meta):
        # MLA：~32K 上下文几十 MiB → 按 1MB/32K token 估
        return 1024 * 1024 / 32768

    def _first(keys):
        # 通用键优先（_read_gguf_meta 归一化），再兜底固定列表
        for k in keys:
            if k in meta:
                return meta[k]
        return None

    n_layers = _first(["block_count"] + _LAYER_KEYS)
    n_kv_heads = _first(["head_count_kv"] + _KV_HEAD_KEYS)
    n_heads = _first(["head_count"] + _HEAD_KEYS)
    head_dim = _first(["head_dim"] + _HEAD_DIM_KEYS)
    embed = _first(["embedding_length"] + _EMBED_KEYS)

    if not n_layers or not n_kv_heads:
        return 0.0
    if not head_dim:
        if n_heads and embed:
            head_dim = embed // n_heads
        else:
            return 0.0

    kv_per_token = 2 * n_layers * n_kv_heads * head_dim * 2  # K+V, F16
    return float(kv_per_token) * KV_SAFETY


def compute_safe_params(
    model_path: str,
    vram_total_mb: int | None = None,
    vram_free_mb: int | None = None,
    want_ctx: int = 8192,
    want_parallel: int = 1,
    force_ctx: bool = False,
) -> dict:
    """根据显存算 safe (n_ctx, n_parallel, kv_budget_gb, reason)。

    返回 dict：
      n_ctx: int          自动降档后的 context
      n_parallel: int     并行度（先保 1；富余时给 want_parallel）
      kv_budget_gb: float KV 可用预算
      reason: str         降档说明（"OK" 或 "OOM 风险，自动降档：..."）

    force_ctx=True → 显式指定优先（--ctx-force），完全信任 want_ctx，
    跳过自动降档。用于用户明确知道显存放得下的场景（如 Qwen3.6-35B-A3B
    IQ2_M 实测 256K KV 仅 ~5.4GB，12G 卡放得下；而估算公式对无 head_dim
    字段的 qwen35moe 高估 16 倍会误降档到 8192，2026-09-09 校准）。
    """
    if force_ctx:
        return {"n_ctx": want_ctx,
                "n_parallel": want_parallel if want_parallel > 1 else 1,
                "kv_budget_gb": 0.0,
                "reason": "OK (ctx-force: 显式指定，跳过自动降档)"}

    total_mb, free_mb = probe_vram_mb()
    if vram_total_mb is not None:
        total_mb = vram_total_mb
    if vram_free_mb is not None:
        free_mb = vram_free_mb

    # 无显存信息（非 N 卡/驱动问题）→ 不降档
    if total_mb <= 0 or free_mb <= 0:
        return {"n_ctx": want_ctx, "n_parallel": want_parallel,
                "kv_budget_gb": 0.0, "reason": "OK (no vram info)"}

    free_gb = free_mb / 1024.0
    # KV 预算 = 可用 - 固定开销(非专家+缓冲 2.5GB) - cache 预留(30%)
    # cache 是懒分配+prefill 会吃掉大部分显存，给 KV 留保守份额
    kv_budget_gb = free_gb - FIXED_VRAM_GB - free_gb * 0.30
    if kv_budget_gb <= 0:
        kv_budget_gb = 0.0

    kv_per_token = estimate_kv_bytes_per_token(model_path)
    n_ctx = want_ctx
    n_parallel = want_parallel if want_parallel > 1 else 1
    reason = "OK"

    if kv_per_token > 0 and kv_budget_gb > 0:
        max_tokens = int(kv_budget_gb * 1024 ** 3 / kv_per_token)
        # 降档顺序：先降 parallel（Qwen OOM 主因是 parallel 4 的 KV 翻倍），
        # 再降 ctx。want_ctx × want_parallel 能 fit 则保持。
        need = want_ctx * want_parallel
        if need <= max_tokens:
            pass  # 保持
        else:
            # 尝试保持 want_ctx，降 parallel
            p = 1
            while p <= want_parallel and want_ctx * p <= max_tokens:
                p += 1
            p -= 1
            if p >= 1 and want_ctx * p <= max_tokens:
                n_parallel = p
                if p < want_parallel:
                    reason = (f"OOM 风险：KV 预算 {kv_budget_gb:.1f}GB（~{max_tokens} tok）"
                              f"→ parallel 降档 {want_parallel}→{p}（ctx 保持 {want_ctx}）")
            else:
                # parallel 降到 1 仍不够 → 降 ctx
                n_parallel = 1
                chosen = None
                for level in CTX_LEVELS:
                    if level <= max_tokens:
                        chosen = level
                        break
                if chosen is None:
                    chosen = CTX_LEVELS[-1]
                n_ctx = chosen
                reason = (f"OOM 风险：KV 预算 {kv_budget_gb:.1f}GB（~{max_tokens} tok）"
                          f"→ 降档 parallel {want_parallel}→1 + context {want_ctx}→{chosen}")
    else:
        n_parallel = want_parallel if want_parallel > 1 else 1

    return {"n_ctx": n_ctx, "n_parallel": n_parallel,
            "kv_budget_gb": round(kv_budget_gb, 2), "reason": reason}


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else ""
    if model:
        print(f"kv/ctx: {estimate_kv_bytes_per_token(model):.0f} B/token")
        print(compute_safe_params(model))
    else:
        print("usage: python3 -m moe_l2.vram_adaptive <model.gguf>")
