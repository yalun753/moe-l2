#!/usr/bin/env python3
"""moe-l2 collect — 收集领域路由数据并生成 domain_expert_map.json

模式 A：一次性收集（新模型接入）
  1. 检查 GGUF 兼容性（arch 标签 + tensor 命名）
  2. 用 LLAMA_EXPERT_LOG=1 跑 8 域 prompt
  3. 解析路由日志 → 生成 domain_expert_map.json
  4. 写到固定位置 ~/.moe-l2/maps/<model_id>/

依赖：llama-cli（支持 LLAMA_EXPERT_LOG 的版本）+ gguf-py
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# 固定位置（与产品化方案一致）
DEFAULT_MAPS_DIR = Path.home() / ".moe-l2" / "maps"

# 8 个领域（与 Phase 1 收集一致）
DEFAULT_DOMAINS = [
    "codegen", "debug", "math", "logic",
    "general_qa", "chinese_tech", "creative_write", "translate",
]

# 每领域 3 阶段 prompt（短/追问/长尾）
DEFAULT_PROMPTS = {
    "codegen": ["Write a Python function to sort a list", "Now add error handling", "Write a full web scraper with retries"],
    "debug": ["Fix this bug: for i in range(10) print(i)", "What caused this segfault?", "Debug a memory leak in this C code"],
    "math": ["Calculate 15 * 37", "What is the derivative of x^3?", "Solve this integral step by step"],
    "logic": ["If A implies B and B implies C, what follows?", "Solve this syllogism", "Design a truth table for XOR"],
    "general_qa": ["What is machine learning", "Explain photosynthesis briefly", "What are the benefits of reading?"],
    "chinese_tech": ["什么是神经网络？简单解释一下", "NAS 怎么搭建？", "Docker 容器和虚拟机有什么区别？"],
    "creative_write": ["Write a poem about the ocean", "Write a short story opening", "Rewrite this paragraph more vividly"],
    "translate": ["Translate 'hello world' to French", "Translate this paragraph to Chinese", "Translate this idiom into English"],
}

MODEL_ID_MAX_LOOKUP = 16  # 兼容性检查时最多读的 tensor 名


def model_id_from_path(model_path: str) -> str:
    """从 GGUF 文件名生成 model_id"""
    name = os.path.basename(model_path)
    return name.replace(".gguf", "").replace(".", "-")


def check_compatibility(model_path: str) -> dict:
    """检查 GGUF 兼容性：直接看文件内容，不依赖架构名白名单。

    [2026-08-13 重构] 不再维护 moe_archs / expert_count key 白名单——
    用户可能用任意模型（任何架构），逐个加白名单永远补不完。
    判定完全从文件 tensor 推断：
      - 是不是 MoE：有 ffn_*_exps.weight（现代）或 ffn_*.N.weight（旧格式）tensor
      - 专家数：ffn_gate_exps 的 shape 维度推断
      - 层数：tensor 名 blk.N. 前缀推断
    架构名（general.architecture）只用于日志展示，不参与判定。
    """
    result = {"ok": False, "arch": None, "has_exps": False, "has_indexed": False, "n_experts": 0, "n_layers": 0, "sample_tensors": []}

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
        from gguf import GGUFReader
        reader = GGUFReader(model_path)
    except Exception as e:
        result["error"] = f"gguf-py 读取失败: {e}"
        return result

    # arch 标签（仅展示，不参与判定）
    try:
        arch_field = reader.get_field("general.architecture")
        arch = bytes(arch_field.parts[-1]).decode("utf-8", "replace").strip("\x00")
        result["arch"] = arch
    except Exception:
        result["arch"] = "?"

    # tensor 扫描：MoE 判定 + 专家数/层数推断（全部来自文件本身）
    sample = []
    has_exps = False
    has_indexed = False
    n_experts_from_tensors = 0
    n_layers_from_tensors = 0
    for t in reader.tensors:
        name = t.name
        if len(sample) < MODEL_ID_MAX_LOOKUP:
            sample.append(name)
        # 现代格式：blk.N.ffn_gate_exps.weight / ffn_up_exps / ffn_down_exps
        if re.search(r"ffn_(gate|up|down)_exps", name):
            has_exps = True
            if "gate" in name:
                try:
                    n_experts_from_tensors = max(n_experts_from_tensors, int(t.shape[-1]))
                except Exception:
                    pass
        # 旧格式：blk.N.ffn_gate.M.weight（带专家序号）
        m_indexed = re.search(r"ffn_(gate|up|down)\.(\d+)\.weight", name)
        if m_indexed:
            has_indexed = True
            try:
                n_experts_from_tensors = max(n_experts_from_tensors, int(m_indexed.group(2)) + 1)
            except Exception:
                pass
        # 层数：blk.N. 前缀
        m_blk = re.match(r"blk\.(\d+)\.", name)
        if m_blk:
            try:
                n_layers_from_tensors = max(n_layers_from_tensors, int(m_blk.group(1)) + 1)
            except Exception:
                pass

    result["has_exps"] = has_exps
    result["has_indexed"] = has_indexed
    result["sample_tensors"] = sample[:8]
    result["n_experts"] = n_experts_from_tensors
    result["n_layers"] = n_layers_from_tensors

    # 判定：完全基于文件内容
    if not has_exps and not has_indexed:
        result["reason"] = "文件中没有 MoE 专家 tensor（ffn_*_exps / ffn_*.N.weight）——不是 MoE 模型"
        return result
    if has_indexed and not has_exps:
        result["reason"] = "旧格式专家 tensor（ffn_gate.N.weight，无 _exps）——llama.cpp 新版本不识别，跳过"
        return result
    result["ok"] = True
    result["reason"] = "OK"
    return result


def collect_routing(llama_cli: str, model_path: str, prompt: str, n_tokens: int, timeout: int = 300) -> list:
    """跑单条 prompt，返回 EXPERT 行列表

    注意：llama-cli 在非 TTY 下输出全部 EXPERT 数据后进程可能不退出（等待
    stdin/CUDA 收尾）。因此输出重定向到临时文件，超时 kill 后从文件读回
    已收集的数据，不丢 EXPERT 行。
    """
    env = os.environ.copy()
    env["LLAMA_EXPERT_LOG"] = "1"
    if os.name != "nt":
        # Linux: point dynamic linker at the bundled .so files next to llama-cli.
        # Windows: DLLs next to llama-cli.exe are auto-loaded.
        lib_dir = os.path.dirname(llama_cli)
        env["LD_LIBRARY_PATH"] = lib_dir + (":" + env.get("LD_LIBRARY_PATH", "") if env.get("LD_LIBRARY_PATH") else "")

    cmd = [llama_cli, "-m", model_path, "-p", prompt, "-n", str(n_tokens),
           "--no-display-prompt", "-c", "2048", "-t", "16", "--no-warmup"]

    tmp = tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False)
    tmp_path = tmp.name
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=tmp, stderr=subprocess.STDOUT, env=env,
                                stdin=subprocess.DEVNULL)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            # 数据已输出但进程没退出 —— 从文件读回
        tmp.close()
        with open(tmp_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        lines = [line for line in content.split("\n") if "EXPERT|" in line]
        if not lines and proc.returncode != 0:
            # 完全失败（模型加载错误等），打印最后几行帮助排查
            err_tail = "\n".join(content.split("\n")[-5:])
            print(f"    ⚠️ llama-cli 退出码 {proc.returncode}: {err_tail[:200]}")
        return lines
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def parse_expert_lines(lines: list) -> dict:
    """解析 EXPERT|L<layer>|T<token>: [id,id...] [id,id...] 为逐层专家频率"""
    layer_freq = defaultdict(lambda: defaultdict(int))
    layer_slots = defaultdict(int)
    n_expert_used = 0
    for line in lines:
        m = re.match(r"EXPERT\|L(\d+)\|T(\d+): (.*)", line.strip())
        if not m:
            continue
        layer = int(m.group(1))
        rest = m.group(3)
        for s in re.findall(r"\[([0-9, ]+)\]", rest):
            ids = [int(x) for x in s.split(",") if x.strip()]
            n_expert_used = max(n_expert_used, len(ids))
            for e in ids:
                layer_freq[layer][e] += 1
                layer_slots[layer] += 1
    return layer_freq, layer_slots, n_expert_used


def build_domain_map(all_freq: dict, n_layers: int, model_path: str, n_experts: int, n_expert_used: int) -> dict:
    """从各领域频率数据生成 domain_expert_map.json"""
    model_id = model_id_from_path(model_path)
    domains = {}
    for domain, layer_freq in all_freq.items():
        per_layer_preferred = {}
        per_layer_top = {}
        total_lines = 0
        total_slots = 0
        for L in range(n_layers):
            freq = layer_freq.get(L, {})
            if not freq:
                per_layer_preferred[str(L)] = []
                per_layer_top[str(L)] = []
                continue
            # 按频率排序（高频在前）
            ranked = [e for e, _ in sorted(freq.items(), key=lambda x: -x[1])]
            per_layer_preferred[str(L)] = ranked
            per_layer_top[str(L)] = ranked[:n_expert_used * 2]  # top-k 取 2 倍激活数
            total_lines += sum(freq.values())
            total_slots += len(ranked)
        domains[domain] = {
            "per_layer_domain_preferred": per_layer_preferred,
            "per_layer_top": per_layer_top,
            "total_lines": total_lines,
            "total_slots": total_slots,
        }

    return {
        "model": model_id,
        "generated_from": model_path,
        "num_experts": n_experts,
        "num_layers": n_layers,
        "experts_per_token": n_expert_used,
        "top_k_per_layer": n_expert_used * 2,
        "backbone_experts": [],
        "layer_specificity": {},
        "domains": domains,
    }


def cmd_collect(args):
    """moe-l2 collect --model <gguf> [--llama-cli <path>] [--output <dir>]"""
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        return 1

    print(f"moe-l2 {__import__('moe_l2').__version__} — collect routing data")
    print(f"  model:  {model_path}")

    # 1. 兼容性检查
    print("\n[1/4] 检查模型兼容性...")
    compat = check_compatibility(model_path)
    if compat.get("error"):
        print(f"  ❌ {compat['error']}")
        return 1
    print(f"  arch:   {compat['arch']}")
    print(f"  tensor: {'✅ _exps 现代格式' if compat['has_exps'] else '❌ 缺 _exps 后缀'}")
    print(f"  experts: {compat['n_experts']}, layers: {compat['n_layers']}")
    if not compat["ok"]:
        print(f"  ❌ 不兼容: {compat.get('reason')}")
        print("     建议下载现代格式 GGUF（bartowski / unsloth / mradermacher 最新版）")
        return 1
    n_experts = compat["n_experts"]
    n_layers = compat["n_layers"]

    # 2. 定位 llama-cli
    llama_cli = args.llama_cli or os.environ.get("LLAMA_CLI")
    if not llama_cli or not os.path.exists(llama_cli):
        print("\n❌ 找不到 llama-cli（需支持 LLAMA_EXPERT_LOG 的版本）")
        print("   用 --llama-cli 指定，或设置 LLAMA_CLI 环境变量")
        return 1
    print(f"  llama-cli: {llama_cli}")

    # 3. 收集路由数据
    domains = args.domains or DEFAULT_DOMAINS
    print(f"\n[2/4] 收集 {len(domains)} 个领域路由数据（LLAMA_EXPERT_LOG=1）...")
    all_freq = {}
    n_expert_used = 0
    for domain in domains:
        print(f"  [{domain}]")
        prompts = DEFAULT_PROMPTS.get(domain, ["General question"])
        domain_freq = defaultdict(lambda: defaultdict(int))
        for i, prompt in enumerate(prompts[: args.stages]):
            print(f"    prompt {i+1}: {prompt[:50]}...")
            lines = collect_routing(llama_cli, model_path, prompt, args.tokens, args.timeout)
            parsed_freq, _, used = parse_expert_lines(lines)
            for L, freq in parsed_freq.items():
                for e, c in freq.items():
                    domain_freq[L][e] += c
            n_expert_used = max(n_expert_used, used)
            print(f"      → {len(lines)} 行")
        all_freq[domain] = domain_freq

    # 4. 生成映射表
    print("\n[3/4] 生成 domain_expert_map.json...")
    domain_map = build_domain_map(all_freq, n_layers, model_path, n_experts, n_expert_used)

    # 5. 写固定位置
    maps_dir = Path(args.output) if args.output else DEFAULT_MAPS_DIR
    model_id = model_id_from_path(model_path)
    out_dir = maps_dir / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "domain_expert_map.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(domain_map, f, ensure_ascii=False, indent=1)
    print(f"\n[4/4] ✅ 已写入 {out_file}")

    # meta.json
    meta = {
        "model": model_id,
        "source": model_path,
        "n_experts": n_experts,
        "n_layers": n_layers,
        "domains_collected": len(all_freq),
        "expert_lines": sum(1 for _ in []),
        "mode": "collect",
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    # 统计
    total_slots = sum(dm["total_slots"] for dm in domain_map["domains"].values())
    print(f"  模型: {model_id}")
    print(f"  领域: {len(all_freq)} 个")
    print(f"  每层专家槽位合计: {total_slots}")
    print(f"  L0a 启动时从 {DEFAULT_MAPS_DIR} 读取即可生效")
    return 0


def main():
    parser = argparse.ArgumentParser(prog="moe-l2-collect", description="Collect MoE routing data for domain→expert map")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--llama-cli", default=None, help="Path to llama-cli (supports LLAMA_EXPERT_LOG)")
    parser.add_argument("--output", default=None, help="Output maps directory (default: ~/.moe-l2/maps)")
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS, help="Domains to collect (default: all 8)")
    parser.add_argument("--stages", type=int, default=3, help="Prompts per domain (default: 3)")
    parser.add_argument("--tokens", type=int, default=20, help="Generation tokens per prompt (default: 20)")
    parser.add_argument("--timeout", type=int, default=300, help="Per-prompt timeout seconds (default: 300)")
    args = parser.parse_args()
    sys.exit(cmd_collect(args))


if __name__ == "__main__":
    main()
