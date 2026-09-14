"""
moe_l2/router_table.py — 模型路由表：按模型选择 + 热门领域自适应 + 缺失自动收集

设计（2026-08-13 用户拍板，医院比喻）：
  医院 = 模型，科室 = 领域，医生 = 专家
  - 每个模型一个路由表（有 → 加载；无 → 自动收集生成）
  - 热门领域（使用频率高）常驻：数量 N 由显存预算决定
  - 热门领域内的 top-k 专家进 GPU cache：k 由覆盖率目标决定（默认 90% → top-75）
  - 冷门领域走 SSD on-demand 兜底（不预填充）
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

# 覆盖率曲线（Qwen3.6-35B-A3B 实测，8 领域均值，2026-08-13 domain_router_coverage.py）
# 目标覆盖率 → top-k（每领域每层保留的专家数）
COVERAGE_TABLE = {
    0.80: 30,
    0.85: 50,
    0.90: 75,   # 默认目标（用户确认）
    0.95: 100,
}

# 实测显存系数：每 1000 cache slots 占多少 GB（含 KV/路由表/graph overhead）
# 来源：11913 slots → 9.8GB（2026-08-13 2080Ti 实测）
VRAM_GB_PER_KSLOPS = 0.82

# 默认显存预算比例：分配给专家 cache 的显存 = 总显存 × 比例（剩余给模型+KV）
DEFAULT_VRAM_BUDGET_RATIO = 0.60


def model_id_from_path(model_path: str) -> str:
    """从 GGUF 文件名提取 model_id（与 collect.py 规则一致）。"""
    name = os.path.basename(model_path)
    return name.replace(".gguf", "").replace(".", "-")


def _parse_router_map_json(path: Path) -> dict:
    """解析路由表 JSON → {layer: [experts]}。兼容 per_layer_domain_preferred / layers 格式。

    2026-08-13 重构：表格式变为 {layer: [experts]}（已按热门领域+top-k 收敛），
    旧格式（domains.<domain>.per_layer_domain_preferred）按领域并集兼容。
    """
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    # 新格式：顶层直接是 {layer: [...]} 或 {"layers": {...}}
    if "layers" in d and isinstance(d["layers"], dict):
        return {int(k): list(v) for k, v in d["layers"].items()}
    # 新格式2：顶层直接数字键
    if all(str(k).isdigit() for k in d.keys()):
        return {int(k): list(v) for k, v in d.items()}

    # 旧格式：domains.<domain>.per_layer_domain_preferred → 按层并集
    layers: dict[int, set] = {}

    def collect(obj) -> bool:
        if isinstance(obj, dict):
            for key in ("per_layer_domain_preferred", "layers", "per_layer"):
                if key in obj and isinstance(obj[key], dict):
                    for lk, experts in obj[key].items():
                        layers.setdefault(int(lk), set()).update(experts)
                    return True
            if "domains" in obj and isinstance(obj["domains"], dict):
                for domdata in obj["domains"].values():
                    collect(domdata)
                return True
        return False

    collect(d)
    return {k: sorted(v) for k, v in layers.items()}


def find_router_table(model_path: str, data_dir: Path | None = None) -> Path | None:
    """查找当前模型的路由表，找不到返回 None。

    [2026-08-13 设计变更] 静态表退役：只认 flywheel 动态表——它是用户真实
    使用数据收敛的用户专属路由表。无表时由调用方触发冷启动采集（通用样本 → 首张表）。

    [2026-08-13 flywheel B] flywheel 表按模型分文件：
    domain_router_map_flywheel_{model_id}.json（各模型独立收敛，互不干扰）。
    不再读取无后缀单文件 domain_router_map_flywheel.json（旧版产物，退役）。
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    model_id = model_id_from_path(model_path)
    # flywheel 表：当前模型的动态收敛表（热度榜 + 领域专家），优先
    flywheel = data_dir / f"domain_router_map_flywheel_{model_id}.json"
    if flywheel.exists():
        return flywheel
    # 兜底：模型专属静态表（早期版本产物，向后兼容）
    exact = data_dir / f"domain_router_map_{model_id}.json"
    if exact.exists():
        return exact
    return None


def top_k_for_coverage(target: float = 0.90) -> int:
    """按覆盖率目标查表得 top-k。目标不在表内时向上取最近的档。"""
    best = 100
    for cov, k in sorted(COVERAGE_TABLE.items()):
        if cov >= target:
            return k
        best = k
    return best


def compute_budget(data_dir: Path | None = None, vram_budget_gb: float | None = None,
                   vram_total_gb: float | None = None) -> float:
    """显存预算（GB）：显式参数 > 环境变量 > 自动检测总显存 × 比例。"""
    if vram_budget_gb is not None:
        return vram_budget_gb
    if os.environ.get("MOE_L2_VRAM_BUDGET"):
        try:
            return float(os.environ["MOE_L2_VRAM_BUDGET"])
        except ValueError:
            pass
    if vram_total_gb is None:
        vram_total_gb = _detect_vram_gb()
    return vram_total_gb * DEFAULT_VRAM_BUDGET_RATIO


def _detect_vram_gb() -> float:
    """自动检测 GPU 总显存（GB）。用 nvidia-smi，失败默认 8GB。"""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if out:
            return float(out.splitlines()[0]) / 1024.0
    except Exception:
        pass
    return 8.0


def auto_collect(model_path: str, llama_cli: str, data_dir: Path | None = None,
                 tokens: int = 20, timeout: int = 120,
                 coverage_target: float = 0.90) -> Path | None:
    """模型无路由表时自动收集：跑 3 条通用样本（覆盖不同领域）→ 统计每层专家激活 →
    按覆盖率取 top-k。

    返回生成的路由表路径；失败返回 None（不阻塞启动，降级为无路由表）。
    """
    from moe_l2.collect import check_compatibility, collect_routing, parse_expert_lines

    print("  [auto-collect] 首次使用该模型，正在生成路由表（约 2-3 分钟，仅首次）...")
    model_path = str(model_path)
    compat = check_compatibility(model_path)
    if compat.get("error") or not compat.get("ok"):
        print(f"  [auto-collect] ⚠️ 模型不兼容，跳过：{compat.get('error') or compat.get('reason')}")
        return None

    # 通用样本：覆盖 3 个不同领域（问答/创作/代码），让首张表尽量中性
    prompts = [
        "Explain how computers work in a few sentences.",      # general_qa
        "Write a short story about a robot learning to paint.",  # creative_write
        "Write a Python function that sorts a list of numbers.",  # codegen
    ]
    layer_freq: dict[int, Counter] = defaultdict(Counter)
    for i, prompt in enumerate(prompts):
        print(f"  [auto-collect] 样本 {i+1}/{len(prompts)}: {prompt[:40]}...")
        lines = collect_routing(llama_cli, model_path, prompt, tokens, timeout)
        parsed, _, _ = parse_expert_lines(lines)
        n = sum(len(v) for v in parsed.values())
        print(f"  [auto-collect]   收集到 {n} 条专家激活")
        for L, freq in parsed.items():
            for e, c in freq.items():
                layer_freq[L][e] += c

    if not layer_freq:
        print("  [auto-collect] ⚠️ 未收集到路由数据（llama-cli 无 EXPERT 输出）")
        return None

    k = top_k_for_coverage(coverage_target)
    per_layer = {
        str(L): [e for e, _ in freq.most_common(k)]
        for L, freq in sorted(layer_freq.items())
    }

    table = {
        "description": f"auto-generated by moe-l2 (first-run collect, top-{k} per layer, "
                       f"coverage target {coverage_target:.0%})",
        "model": model_id_from_path(model_path),
        "top_k": k,
        "layers": per_layer,
    }
    data_dir = data_dir or DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"domain_router_map_{model_id_from_path(model_path)}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=1)
    print(f"  [auto-collect] ✅ 已生成 {out} (top-{k}/层)")
    return out


def filter_hot_domains(table_path: Path, hot_domains: list[str] | None = None,
                       coverage_target: float = 0.90,
                       n_hot: int | None = None) -> Path | None:
    """从旧格式路由表（domains.<domain>.per_layer...）中只保留热门领域，按覆盖率收敛 top-k。

    n_hot：热门领域数上限（热度榜前 N）；None 表示不限制（全领域/显式列表）。
    返回新格式临时表路径；旧表已是新格式时原样返回。
    """
    with open(table_path, encoding="utf-8") as f:
        d = json.load(f)

    # 已是新格式（顶层 layers 或数字键）→ 直接返回
    if "layers" in d or all(str(k).isdigit() for k in d.keys()):
        return table_path

    domains = d.get("domains", {})
    if not domains:
        return table_path

    # 确定热门领域优先级：
    #   1. 显式指定 hot_domains
    #   2. 表内 dom_freq 热度榜（flywheel 写入）
    #   3. 旧表顺序（兜底，全领域）
    if hot_domains is None:
        dom_freq = d.get("dom_freq", {})
        if dom_freq:
            ranked = [x for x, _ in sorted(dom_freq.items(), key=lambda kv: -kv[1])]
            hot_domains = [x for x in ranked if x in domains]
        else:
            hot_domains = list(domains.keys())
    # 热门领域数上限
    if n_hot is not None and n_hot > 0:
        hot_domains = hot_domains[:n_hot]
    k = top_k_for_coverage(coverage_target)

    layers: dict[int, Counter] = defaultdict(Counter)
    for dom in hot_domains:
        domdata = domains.get(dom)
        if not domdata:
            continue
        plist = domdata.get("per_layer_domain_preferred", {})
        for lk, experts in plist.items():
            # 旧表是 top-100，这里截断到 k
            # 按"领域覆盖数"加权：同一专家出现在越多热门领域，得分越高
            # （跨领域通用专家优先保留；单领域专家也能进，只是排后面）
            for e in list(experts)[:k]:
                layers[int(lk)][e] += 1

    # [A-fix 2026-08-15] 不再按 ID 排序截断（会把 union 截成 ID 最小的 k 个，
    # 与覆盖率意图相反）；改为按领域覆盖数降序取 top-k（Counter.most_common）。
    per_layer = {
        str(L): [e for e, _ in layers[L].most_common(k)]
        for L in sorted(layers)
    }
    new_table = {
        "description": f"hot-domain filtered (domains={hot_domains}, top-{k}/layer, "
                       f"coverage target {coverage_target:.0%})",
        "model": d.get("model", ""),
        "top_k": k,
        "layers": per_layer,
    }
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, prefix="moe-l2-router-")
    json.dump(new_table, tmp, ensure_ascii=False, indent=1)
    tmp.close()
    return Path(tmp.name)


def build_router_map_file(model_path: str, router_top_k: int = 75,
                          data_dir: Path | None = None,
                          llama_cli: str | None = None,
                          coverage_target: float = 0.90,
                          vram_budget_gb: float | None = None,
                          hot_domains: list[str] | None = None) -> str | None:
    """start 时调用的主入口：
    1. 按模型找表（有 → 用）
    2. 无表 → 自动收集 → 生成
    3. 显存预算 → 热门领域数 N（N=0 时纯 on-demand）
    4. 热门领域（热度榜前 N）→ 覆盖率 top-k 截断
    5. 生成 router.map 临时文件（C++ MOE_L2_ROUTER_FILE 格式）
    返回 router.map 路径；失败返回 None（不阻塞启动）。
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    budget = compute_budget(data_dir, vram_budget_gb)

    # ---- N 计算：显存预算 → 热门领域数 ----
    # 每领域成本 = top-k(覆盖率) × 层数 × 3 tensor → 显存
    # 注意：用"最小可跑档"（每层 30 专家）而非全量 top-k 档 ——
    # 预算不足时会按预算收敛 top-k（如 4.8GB → 48/层），而不是直接 N=0。
    k_coverage = top_k_for_coverage(coverage_target)
    n_layers_est = 40
    table_path = find_router_table(model_path, data_dir)
    if table_path is not None:
        try:
            with open(table_path, encoding="utf-8") as _f:
                _d = json.load(_f)
            if "layers" in _d:
                n_layers_est = len(_d["layers"])
            elif "domains" in _d:
                for _dd in _d["domains"].values():
                    pl = _dd.get("per_layer_domain_preferred", {})
                    if pl:
                        n_layers_est = len(pl)
                        break
        except Exception:
            pass
    min_domain_vram = 30 * n_layers_est * 3 / 1000.0 * VRAM_GB_PER_KSLOPS
    full_domain_vram = k_coverage * n_layers_est * 3 / 1000.0 * VRAM_GB_PER_KSLOPS
    # N = 预算能养几个领域（按最小可跑档算，避免预算刚好差一点就 N=0）
    n_hot = max(0, int(budget / min_domain_vram)) if min_domain_vram > 0 else 1

    if n_hot == 0:
        print(f"  [router] 显存预算 {budget:.1f}GB < 最小档 {min_domain_vram:.1f}GB"
              f" → N=0 跳过 prefill，纯 on-demand 模式（小显存卡可跑）")
        return None
    print(f"  [router] 显存预算 {budget:.1f}GB（单领域最小 {min_domain_vram:.1f}GB / "
          f"全量 {full_domain_vram:.1f}GB）→ 热门领域 N={n_hot}")

    # ---- 找表 / 自动收集 ----
    if table_path is None:
        if llama_cli is None:
            bin_dir = Path(__file__).resolve().parent / "bin"
            # Windows: the bundled binary is llama-cli.exe, not llama-cli
            _cli_cands = (
                ["llama-cli.exe", "llama-cli"]
                if os.name == "nt"
                else ["llama-cli", "llama-cli.exe"]
            )
            for _c in _cli_cands:
                if (bin_dir / _c).exists():
                    llama_cli = str(bin_dir / _c)
                    break
        if llama_cli:
            table_path = auto_collect(model_path, llama_cli, data_dir, coverage_target=coverage_target)
        else:
            print("  [router] ⚠️ 无路由表且无 llama-cli，跳过 selective pin（降级 on-demand）")
            return None
    if table_path is None:
        return None

    # ---- 热门领域截断：显式 hot_domains 优先，否则热度榜前 N ----
    if hot_domains is not None:
        hot_domains = hot_domains[:n_hot] if n_hot > 0 else []
    filtered = filter_hot_domains(table_path, hot_domains, coverage_target, n_hot=n_hot)
    if filtered is None:
        return None

    layers = _parse_router_map_json(filtered)
    if not layers:
        print("  [router] ⚠️ 路由表为空，跳过 selective pin")
        return None

    # 显存预算 → 每层 top-k 上限（覆盖率档之外再按预算收敛）
    n_layers = len(layers)
    exp_per_layer = max(len(v) for v in layers.values())
    slots_est = exp_per_layer * n_layers * 3
    vram_est = slots_est / 1000.0 * VRAM_GB_PER_KSLOPS
    if vram_est > budget and exp_per_layer > 30:
        shrink = int(exp_per_layer * budget / vram_est)
        shrink = max(30, min(shrink, exp_per_layer))
        # N=0 档：预算连最小档（每层 30 专家）都撑不起 → 跳过 prefill，
        # 全部走 on-demand（SSD/RAM 现拷），小显存卡也能跑。
        if shrink <= 30 and budget < 30 * n_layers * 3 / 1000.0 * VRAM_GB_PER_KSLOPS:
            print(f"  [router] 显存预算 {budget:.1f}GB 不足以支撑专家 cache（需 "
                  f"{30 * n_layers * 3 / 1000.0 * VRAM_GB_PER_KSLOPS:.1f}GB 最小档），"
                  f"N=0 跳过 prefill → 纯 on-demand 模式")
            return None
        print(f"  [router] 显存预算 {budget:.1f}GB，预估 {vram_est:.1f}GB → top-k 收敛到 {shrink}")
        for L in layers:
            layers[L] = layers[L][:shrink]

    tmp = tempfile.NamedTemporaryFile("w", suffix=".router.map", delete=False, prefix="moe-l2-router-")
    # [moe-l2 2026-08-14 cache 容量联动] 第一行元数据：# EXPERT_TOTAL <n>
    # C++ A3 cache 用它算 n_slots（= 路由表选中专家数 × 3 张量），替代
    # fraction × 全模型专家空间公式 —— 回到 TOP N / TOP K 设计：cache 只服务
    # 路由表选中的专家，容量 = 选中数 × 3，不再被全模型空间放大。
    # 先统计总数（写元数据行时就得是最终值）
    _layer_experts = {
        L: sorted(layers[L])[:router_top_k] for L in sorted(layers.keys())
    }
    total_experts = sum(len(v) for v in _layer_experts.values())
    tmp.write(f"# EXPERT_TOTAL {total_experts}\n")
    for layer, experts in _layer_experts.items():
        tmp.write(f"{layer} " + " ".join(str(e) for e in experts) + "\n")
    tmp.close()
    print(f"  selective pin: router map = {tmp.name} ({len(layers)} layers, "
          f"{total_experts} experts, top-k≤{router_top_k})")
    return tmp.name
