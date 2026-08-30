"""moe-l2 CLI entry point.

Usage:
    moe-l2 start [--model PATH] [--l2-size SIZE] [--port PORT] [--gpu]
    moe-l2 stats [--port PORT]
    moe-l2 download-bins [--release TAG]
    moe-l2 doctor
    moe-l2 model list
    moe-l2 model download --model NAME [--dir DIR]
    moe-l2 --help
"""

import argparse
import logging
import os
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Optional

import httpx

from . import __version__
from .cache import L2Cache
from .proxy import DEFAULT_HOST, DEFAULT_PORT, LLAMA_SERVER_BASE, start_proxy

logger = logging.getLogger("moe-l2")

# Common GGUF search paths
_GGUF_PATHS = [
    "/opt/data/models/Qwen2.5-MOE-2X1.5B-Q2_k.gguf",
    "/opt/data/models/DeepSeek-V2-Lite-Chat-Uncensored.Q2_K.gguf",
    "/opt/data/models/*.gguf",
]

# GitHub release info
_GITHUB_REPO = "yalun753/moe-l2"
_BINS_ASSET_URL = (
    "https://github.com/{repo}/releases/download/{tag}/llama_bins.tar.gz"
)
_DEFAULT_BINS_TAG = "bins-v0.7.0"

# Where the bundled llama-server lives (relative to this file)
_BUNDLE_DIR = Path(__file__).resolve().parent / "bin"
_LLAMA_SERVER_PATH = _BUNDLE_DIR / "llama-server"
_GPU_PORT = 11436  # llama-server listens here; proxy forwards to it

# Model catalog for `moe-l2 model download` (hf-mirror.com mirrors HuggingFace)
# name -> (repo_id, [files])  — files are relative to repo root
_MODEL_CATALOG = {
    "qwen3.6-35b": {
        "repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "files": ["Qwen3.6-35B-A3B-UD-IQ2_M.gguf"],
        "size": "~11.5 GB (IQ2_M, 35B-A3B — recommended demo)",
    },
    "deepseek-v2-lite": {
        "repo": "mradermacher/DeepSeek-V2-Lite-Chat-Uncensored-GGUF",
        "files": ["DeepSeek-V2-Lite-Chat-Uncensored.Q2_K.gguf"],
        "size": "~6.1 GB (Q2_K, 16B MoE — smallest footprint)",
    },
    "qwen3-235b": {
        "repo": "unsloth/Qwen3-235B-A22B-GGUF",
        "files": [
            "Q2_K/Qwen3-235B-A22B-Q2_K-00001-of-00002.gguf",
            "Q2_K/Qwen3-235B-A22B-Q2_K-00002-of-00002.gguf",
        ],
        "size": "~81.7 GB (Q2_K, 235B-A22B — flagship, needs 64GB+ RAM)",
    },
}
_HF_MIRROR = "https://hf-mirror.com"


def _hf_url(repo: str, path: str) -> str:
    return f"{_HF_MIRROR}/{repo}/resolve/main/{path}"


def _find_gguf(path_hint: str) -> str | None:
    """Find a GGUF model file, trying path_hint first, then common paths."""
    if path_hint and path_hint != "auto":
        p = Path(path_hint)
        if p.exists():
            return str(p)
        logger.warning("Specified model not found: %s", path_hint)

    # Try common paths
    for pattern in _GGUF_PATHS:
        matches = sorted(Path("/opt/data/models").glob("*.gguf"))
        if matches:
            return str(matches[0])

    return None


def _parse_l2_size(size_str: str, expert_size: int) -> int:
    """Parse '4GB', '2GB', etc. into slots_per_layer."""
    size_str = size_str.strip().upper()
    if size_str.endswith("GB"):
        total_bytes = float(size_str[:-2]) * (1024**3)
    elif size_str.endswith("MB"):
        total_bytes = float(size_str[:-2]) * (1024**2)
    else:
        raise ValueError(f"Unrecognized size format: {size_str} (use e.g. 4GB)")

    slots = int(total_bytes / expert_size)
    return max(slots, 1)


def _ensure_bins(tag: str | None = None) -> bool:
    """Download bundled binaries if not present. Returns True if ready."""
    # Self-heal: older releases extract as bin/bin/llama-server (nested prefix).
    if not _LLAMA_SERVER_PATH.exists() and (_BUNDLE_DIR / "bin" / "llama-server").exists():
        _flatten_nested_bins()
    if _LLAMA_SERVER_PATH.exists():
        return True
    print(f"  [download-bins] Binary not found at {_BUNDLE_DIR}")
    print("  Auto-downloading from GitHub Releases...")
    return _download_bins(tag or _DEFAULT_BINS_TAG)


def _flatten_nested_bins() -> None:
    """Move moe_l2/bin/bin/* up one level (older tarballs carry a bin/ prefix)."""
    nested = _BUNDLE_DIR / "bin"
    if not nested.is_dir():
        return
    print("  [download-bins] Detected nested bin/ layout, flattening ...")
    for item in sorted(nested.iterdir()):
        dest = _BUNDLE_DIR / item.name
        if item.is_dir() and dest.exists() and not dest.is_symlink():
            # merge directory (e.g. cuda-libs inside bin/)
            for sub in sorted(item.iterdir()):
                if not (dest / sub.name).exists():
                    sub.rename(dest / sub.name)
            try:
                item.rmdir()  # drop now-empty nested subdir
            except OSError:
                pass
        elif not dest.exists():
            item.rename(dest)
    # drop now-empty nested dir
    try:
        nested.rmdir()
    except OSError:
        pass


def _download_bins(tag: str) -> bool:
    """Download and extract the pre-built llama-server + .so bundle."""
    url = _BINS_ASSET_URL.format(repo=_GITHUB_REPO, tag=tag)
    print(f"  Downloading binaries from {url} ...")
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=120.0)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        print(f"  ERROR: Failed to download: {e}")
        return False

    tgz_path = _BUNDLE_DIR / "llama_bins.tar.gz"
    _BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    tgz_path.write_bytes(resp.content)

    print(f"  Extracting to {_BUNDLE_DIR} ...")
    with tarfile.open(tgz_path, "r:gz") as tf:
        tf.extractall(path=_BUNDLE_DIR)

    tgz_path.unlink()  # cleanup

    # Self-heal: tarballs from bins-v0.2.0 ship a bin/ prefix → bin/bin/llama-server
    _flatten_nested_bins()

    # Verify
    if _LLAMA_SERVER_PATH.exists():
        size = _LLAMA_SERVER_PATH.stat().st_size
        print(f"  OK: llama-server ({size} bytes)")
        return True
    print("  ERROR: llama-server not found after extraction")
    return False


def _start_llama_server(
    model_path: str,
    port: int,
    router_map: str | None = None,
    router_top_k: int = 100,
    n_ctx: int | None = None,
    n_parallel: int | None = None,
) -> subprocess.Popen:
    """Launch bundled llama-server with GPU support (A3 patch enabled).

    Sets LD_LIBRARY_PATH to the bundled .so files.
    Uses mmap default so expert tensors stay in CPU RAM.
    router_map: optional path to selective-pin router map (MOE_L2_ROUTER_FILE).
    When router_map is None, no selective pin (whole-pin default).
    n_ctx/n_parallel: 显式指定则用；None 时启动前探测显存自动算 safe 值
    （防 KV 预分配 OOM 崩溃，2026-08-15 vram_adaptive）。
    """
    if not _LLAMA_SERVER_PATH.exists():
        raise FileNotFoundError(
            f"llama-server not found at {_LLAMA_SERVER_PATH}. "
            "Run 'moe-l2 download-bins' or reinstall moe-l2 with GPU support."
        )

    env = os.environ.copy()
    # Point dynamic linker at bundled .so files (bin + cuda-libs).
    # cuda-libs ships libcudart/libcublas/libcublasLt/libnccl — the target
    # machine may only have a bare NVIDIA driver without the CUDA runtime.
    lib_paths = [str(_BUNDLE_DIR)]
    cuda_libs = _BUNDLE_DIR / "cuda-libs"
    if cuda_libs.is_dir():
        lib_paths.append(str(cuda_libs))
    env["LD_LIBRARY_PATH"] = (
        ":".join(lib_paths) + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    # Force expert MUL_MAT_ID ops onto GPU even at batch=1 (single-token decode).
    # Default GGML_OP_OFFLOAD_MIN_BATCH is 32 (ggml-cuda.cu), so decode would keep
    # experts on CPU. With host-buffer experts (llama-model-loader moe-l2 patch),
    # sched's MoE expert-copy optimization copies only activated experts → GPU fast
    # path. Measured: DS 12.5→37.5 t/s, Qwen 10→46.8 t/s (RTX 4090, 2026-08-02).
    env["GGML_OP_OFFLOAD_MIN_BATCH"] = "1"
    # [moe-l2 2026-08-07] A3 expert cache (2048 槽上限)：热专家缓存在 GPU 显存，
    # 省重复 PCIe 拷贝。实测三模型通用增益：Qwen 50.2 / DS 37.9 / V4 10.1 t/s（4090）。
    env["GGML_CUDA_EXPERT_CACHE"] = "1"
    # 门控在线自适应（P2 ④）：输出实时路由日志（EXPERT|Lx|Ty: [...]），
    # 由 proxy/gate 解析后动态调缓存优先级
    env["LLAMA_EXPERT_LOG"] = "1"
    # [moe-l2 2026-08-10 selective pin] 只 pin 路由表内专家（MOE_L2_ROUTER_FILE），
    # 表外专家走 on-demand fault 兜底。RSS 从 whole-tensor 降到 top-K 专家集。
    # router_map 为 None 时不启用（保持 whole-pin 默认）。
    if router_map:
        env["MOE_L2_ROUTER_FILE"] = router_map
        print(f"  selective pin: router map = {router_map} (top-k={router_top_k})")
    # [moe-l2 2026-08-07 修订] on-demand pin 已编译进二进制（ggml_cuda_expert_pin_host：
    # 首次触碰时合并注册整个专家 tensor，MOEL2_WHOLE_PIN 默认开），无需环境变量。
    # 旧逻辑（GGML_CUDA_REGISTER_HOST=1 按模型大小开关）基于"lazy+REGISTER_HOST=46.7"
    # 的错误前提，实测该组合只有 11.6 t/s，已删除（备份 cli-20260807-register-host-前.py）。
    # 新方案实测：Qwen 50.2 / DS 37.9 / V4 10.1 t/s（4090，GGML_CUDA_EXPERT_CACHE=1）。
    # mmap default: expert tensors stay in CPU RAM (host buffer), GPU cuBLAS computes them on demand

    cmd = [
        str(_LLAMA_SERVER_PATH),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--model", model_path,
        # Reasonable defaults for consumer GPUs
        "-ngl", "99",       # offload non-expert layers to GPU
    ]

    # [moe-l2 2026-08-15 显存自适应] KV 预分配按 parallel × n_ctx，显存不足直接
    # cudaMalloc 崩溃（Qwen cache + parallel 4 在 11GB 卡 OOM 实测）。启动前
    # 探测显存 + 按模型 KV 估算自动降档。显式传入的 n_ctx/n_parallel 作为
    # "期望值"，显存不够时仍会降档（想要 4 路但放不下 → 自动降到能跑的）。
    try:
        from moe_l2.vram_adaptive import compute_safe_params
        params = compute_safe_params(
            model_path,
            want_ctx=n_ctx or 8192,
            want_parallel=n_parallel or 1,
        )
        n_ctx = params["n_ctx"]
        n_parallel = params["n_parallel"]
        if params["reason"] != "OK":
            print(f"  [vram-adaptive] {params['reason']}")
    except Exception as e:  # 自适应失败不阻塞启动，用默认
        print(f"  [vram-adaptive] ⚠️ 自动降档失败，用默认 c8192/parallel1：{e}")
        n_ctx = n_ctx or 8192
        n_parallel = n_parallel or 1

    cmd += ["-c", str(n_ctx)]
    if n_parallel and n_parallel > 1:
        cmd += ["--parallel", str(n_parallel)]

    logger.info("Launching llama-server (A3 GPU): %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


class _GateReaderThread:
    """读 llama-server stderr，把 EXPERT 路由日志喂给 RoutingProfiler。

    守护线程：llama-server 退出或 proxy 停止时自动结束。
    """

    def __init__(self, proc: subprocess.Popen, gate):
        self._proc = proc
        self._gate = gate
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import select as _select
        while not self._stop:
            try:
                r, _, _ = _select.select([self._proc.stderr], [], [], 0.5)
                if not r:
                    continue
                line = self._proc.stderr.readline()
                if not line:
                    break  # stderr 关闭（llama-server 退出）
                self._gate.on_log_line(line.decode(errors="replace"))
            except Exception:
                break

    def stop(self) -> None:
        self._stop = True


def _spawn_gate_reader(proc: subprocess.Popen, gate):
    """启动门控 stderr 采集线程。"""
    return _GateReaderThread(proc, gate)


def _wait_for_llama_server(port: int, timeout: float = 180.0) -> bool:
    """Poll until llama-server responds on /health.

    [moe-l2 2026-08-09] timeout 30→180s：V4 85GB 模型加载约 90s，原 30s 导致
    大模型 start 必失败（TIMEOUT → communicate(5) → kill）。180s 覆盖 V4 加载。
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if resp.status_code < 500:
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(
        prog="moe-l2",
        description="MoE inference L2 hot-cache scheduler",
    )
    parser.add_argument(
        "--version", action="version", version=f"moe-l2 {__version__}"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # moe-l2 start
    start_parser = subparsers.add_parser(
        "start", help="Start the L2 scheduler + proxy"
    )
    start_parser.add_argument(
        "--model",
        default="auto",
        help=(
            "Path to GGUF model file (auto-detect if omitted). "
            "Required for automated n_layers/expert_size detection."
        ),
    )
    start_parser.add_argument(
        "--l2-size",
        type=str,
        default="4GB",
        help="L2 cache size per layer (e.g. 4GB, 2GB). Default: 4GB",
    )
    start_parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Proxy port (default: {DEFAULT_PORT})",
    )
    start_parser.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Enable GPU-accelerated inference with bundled llama-server. "
            "Requires CUDA + NVIDIA GPU. Uses the A3 tiered expert scheduler "
            "to fit large MoE models in limited VRAM."
        ),
    )
    start_parser.add_argument(
        "--backend-url",
        type=str,
        default=None,
        help=(
            "[moe-l2 2026-08-30 remote] Proxy a REMOTE llama-server (e.g. "
            "http://192.168.31.210:11436). Skips local binary/model startup; "
            "proxy + gate (domain prediction + POST /moe-set-domain) still work. "
            "--model is treated as a model_id (e.g. 'qwen') to pick the flywheel "
            "table; default 'qwen'."
        ),
    )
    start_parser.add_argument(
        "--router-map",
        type=str,
        default=None,
        help=(
            "[moe-l2 2026-08-10 selective pin] Path to router map file (format: "
            "'layer expert1 expert2 ...' per line). When set, llama-server only "
            "pins experts listed in the map (RSS = top-K experts, not whole tensor). "
            "If omitted, moe-l2 auto-generates one from the flywheel/static table "
            "with --router-top-k."
        ),
    )
    start_parser.add_argument(
        "--router-top-k",
        type=int,
        default=75,
        help=(
            "[moe-l2 2026-08-13] Max experts per layer for auto-generated router "
            "map (default: 75 = coverage ~90%%). Only used when --router-map is not given."
        ),
    )
    start_parser.add_argument(
        "--coverage-target",
        type=float,
        default=0.90,
        help=(
            "[moe-l2 2026-08-13] Coverage target for auto router map (default: 0.90). "
            "top-k derived from measured coverage curve (80%%->30, 85%%->50, "
            "90%%->75, 95%%->100), then clamped by VRAM budget."
        ),
    )
    start_parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help=(
            "[moe-l2 2026-08-15] Concurrent slots (--parallel N). KV cache is "
            "pre-allocated per slot, so on small GPUs a large N can OOM — "
            "moe-l2 auto-probes VRAM and downgrades parallel/context if needed."
        ),
    )
    start_parser.add_argument(
        "--ctx-size",
        type=int,
        default=8192,
        help=(
            "[moe-l2 2026-08-15] Context size (default: 8192). Auto-downgraded "
            "when VRAM is insufficient (see --parallel)."
        ),
    )

    # moe-l2 stats
    stats_parser = subparsers.add_parser(
        "stats", help="Show live cache statistics from running proxy"
    )
    stats_parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Proxy port (default: {DEFAULT_PORT})",
    )

    # moe-l2 download-bins
    download_parser = subparsers.add_parser(
        "download-bins", help="Download pre-built GPU binaries from GitHub Release"
    )
    download_parser.add_argument(
        "--release",
        type=str,
        default=_DEFAULT_BINS_TAG,
        help=f"GitHub Release tag (default: {_DEFAULT_BINS_TAG})",
    )

    # moe-l2 doctor
    subparsers.add_parser(
        "doctor", help="Environment self-check (GPU, CUDA, Python, disk, binaries)"
    )

    # moe-l2 model list / model download
    model_parser = subparsers.add_parser(
        "model", help="List / download models from hf-mirror.com"
    )
    model_sub = model_parser.add_subparsers(dest="model_command", help="Model sub-commands")
    model_sub.add_parser("list", help="List available models")
    model_dl_parser = model_sub.add_parser("download", help="Download a model")
    model_dl_parser.add_argument("--model", required=True, help="Model name (see 'moe-l2 model list')")
    model_dl_parser.add_argument(
        "--dir", default="~/.moe-l2/models",
        help="Output directory (default: ~/.moe-l2/models)",
    )

    # moe-l2 collect
    collect_parser = subparsers.add_parser(
        "collect", help="Collect MoE routing data → domain_expert_map.json (模式 A)"
    )
    collect_parser.add_argument("--model", required=True, help="Path to GGUF model file")
    collect_parser.add_argument("--llama-cli", default=None, help="Path to llama-cli (supports LLAMA_EXPERT_LOG)")
    collect_parser.add_argument("--output", default=None, help="Output maps dir (default: ~/.moe-l2/maps)")
    collect_parser.add_argument("--domains", nargs="+", default=None, help="Domains to collect (default: all 8)")
    collect_parser.add_argument("--stages", type=int, default=3, help="Prompts per domain (default: 3)")
    collect_parser.add_argument("--tokens", type=int, default=20, help="Gen tokens per prompt (default: 20)")
    collect_parser.add_argument("--timeout", type=int, default=300, help="Per-prompt timeout (default: 300)")

    # moe-l2 embed-map
    embed_parser = subparsers.add_parser(
        "embed-map", help="Embed domain_expert_map.json into a GGUF model (发布形态)"
    )
    embed_parser.add_argument("--model", required=True, help="Path to GGUF model file")
    embed_parser.add_argument("--map", default=None, help="Path to domain_expert_map.json (default: bundled)")
    embed_parser.add_argument("--output", required=True, help="Output GGUF file path")
    embed_parser.add_argument("--keep-original", action="store_true", help="Keep source model (default: delete on success)")

    # moe-l2 version
    subparsers.add_parser(
        "version", help="Show version (alternative to --version)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(message)s",
        )

    if args.command == "start":
        return cmd_start(args)
    elif args.command == "stats":
        return cmd_stats(args)
    elif args.command == "download-bins":
        return cmd_download_bins(args)
    elif args.command == "doctor":
        return cmd_doctor(args)
    elif args.command == "model":
        if args.model_command == "list":
            return cmd_model_list(args)
        elif args.model_command == "download":
            return cmd_model_download(args)
        else:
            print("moe-l2 model: use 'list' or 'download'. Try: moe-l2 model list")
            return 1
    elif args.command == "collect":
        from .collect import cmd_collect
        return cmd_collect(args)
    elif args.command == "embed-map":
        import os as _os

        from .gguf_embed import embed_map
        map_path = args.map or _os.path.join(
            _os.path.dirname(__file__), "data", "domain_expert_map.json"
        )
        out = embed_map(
            args.model, map_path, args.output, keep_original=args.keep_original
        )
        print(f"✅ Embedded domain map → {out}")
        return 0
    elif args.command == "version":
        print(f"moe-l2 {__version__}")
        return 0
    else:
        parser.print_help()
        return 1


def cmd_download_bins(args):
    """Download pre-built GPU binaries from GitHub Release."""
    print(f"moe-l2 {__version__} — downloading GPU binaries")
    print(f"  target:  {_BUNDLE_DIR}")
    ok = _download_bins(args.release)
    return 0 if ok else 1


# ────────────────────────────────────────────────────────────────
# moe-l2 doctor — environment self-check
# ────────────────────────────────────────────────────────────────
def _check_nvidia() -> tuple[bool, str]:
    """Check NVIDIA GPU + driver via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return False, "nvidia-smi not available (no NVIDIA driver?)"
        lines = [line.strip() for line in r.stdout.strip().splitlines() if line.strip()]
        if not lines:
            return False, "nvidia-smi returned no GPU"
        return True, lines[0]
    except FileNotFoundError:
        return False, "nvidia-smi not found on PATH"
    except Exception as e:
        return False, f"nvidia-smi error: {e}"


def _check_cuda_lib() -> tuple[bool, str]:
    """Check libcuda.so is loadable (needed by bundled .so bundle)."""
    import ctypes
    try:
        ctypes.CDLL("libcuda.so.1")
        return True, "libcuda.so.1 found"
    except OSError:
        pass
    try:
        ctypes.CDLL("libcuda.so")
        return True, "libcuda.so found"
    except OSError:
        return False, "libcuda.so not found (install NVIDIA CUDA driver)"


def _check_python() -> tuple[bool, str]:
    v = sys.version_info
    if v >= (3, 10):
        return True, f"Python {v.major}.{v.minor}.{v.micro} (>=3.10 OK)"
    return False, f"Python {v.major}.{v.minor}.{v.micro} (need >=3.10)"


def _check_disk(path: Path) -> tuple[bool, str]:
    # statvfs needs an existing dir; walk up to nearest ancestor
    p = path
    while p and not p.exists():
        p = p.parent
    try:
        st = os.statvfs(str(p))
        free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
        if free_gb >= 5:
            return True, f"{free_gb:.1f} GB free (for {path})"
        return False, f"only {free_gb:.1f} GB free (for {path}, need >=5GB)"
    except OSError:
        return True, f"cannot stat {path} (assuming OK)"


def _check_dynamic_libs() -> tuple[bool, str]:
    """Run ldd on llama-server (with bundled lib paths) and report missing libs.

    The existence check alone is not enough: a binary can be present but fail
    to launch when a shared library (e.g. libnccl.so.2) is missing.
    """
    if not _LLAMA_SERVER_PATH.exists():
        return False, "llama-server missing — run 'moe-l2 download-bins'"
    env = os.environ.copy()
    lib_paths = [str(_BUNDLE_DIR)]
    cuda_libs = _BUNDLE_DIR / "cuda-libs"
    if cuda_libs.is_dir():
        lib_paths.append(str(cuda_libs))
    env["LD_LIBRARY_PATH"] = ":".join(lib_paths) + ":" + env.get("LD_LIBRARY_PATH", "")
    try:
        r = subprocess.run(
            ["ldd", str(_LLAMA_SERVER_PATH)],
            capture_output=True, text=True, timeout=30, env=env,
        )
    except FileNotFoundError:
        return False, "ldd not available on this system"
    missing = [
        line.strip() for line in r.stdout.splitlines()
        if "not found" in line
    ]
    if missing:
        return False, "missing shared libs: " + "; ".join(missing)
    return True, "llama-server dynamic deps OK"


def cmd_doctor(args):
    """Run environment self-check (GPU, CUDA, Python, disk, binaries)."""
    print(f"moe-l2 {__version__} — doctor: environment self-check")
    print("=" * 58)
    checks = [
        ("Python", _check_python()),
        ("NVIDIA GPU", _check_nvidia()),
        ("CUDA library", _check_cuda_lib()),
        ("Disk space", _check_disk(Path.home() / ".moe-l2")),
        ("Bundled binaries", (
            True, "llama-server present" if _LLAMA_SERVER_PATH.exists()
            else "missing — run 'moe-l2 download-bins'"
        )),
        ("Dynamic libraries", _check_dynamic_libs()),
    ]
    ok_count = 0
    for name, (ok, detail) in checks:
        mark = "✅" if ok else "❌"
        print(f"  {mark} {name:<16} {detail}")
        ok_count += int(ok)

    print("=" * 58)
    if ok_count == len(checks):
        print("  All checks passed. Run:")
        print("    moe-l2 start --model <your.gguf> --gpu")
        return 0
    print(f"  {len(checks) - ok_count} check(s) failed — see above for fixes.")
    return 1


# ────────────────────────────────────────────────────────────────
# moe-l2 model — download models from hf-mirror.com
# ────────────────────────────────────────────────────────────────
def _download_file(url: str, dest: Path) -> bool:
    """Stream a file with resume support. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    try:
        with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                          timeout=60.0) as resp:
            if existing and resp.status_code == 416:
                part.rename(dest)  # already complete
                print(f"  ✅ {dest.name} (already complete)")
                return True
            if resp.status_code not in (200, 206):
                print(f"  ❌ HTTP {resp.status_code} for {url}")
                return False
            total = int(resp.headers.get("content-length", 0)) + existing
            written = existing
            mode = "ab" if existing else "wb"
            with open(part, mode) as f:
                for chunk in resp.iter_bytes(1024 * 1024):
                    f.write(chunk)
                    written += len(chunk)
                    if total and written % (8 * 1024 * 1024) < 1024 * 1024:
                        pct = written / total * 100
                        print(f"\r  {dest.name}: {written/1e9:.1f}/{total/1e9:.1f} GB ({pct:.1f}%)", end="", flush=True)
            print()
    except Exception as e:
        print(f"  ❌ download failed: {e}")
        return False
    part.rename(dest)
    return True


def cmd_model_list(args):
    print(f"moe-l2 {__version__} — available models (via hf-mirror.com)")
    print("=" * 58)
    for name, meta in _MODEL_CATALOG.items():
        print(f"  {name:<18} {meta['size']}")
        print(f"    repo: {meta['repo']}")
    print("=" * 58)
    print("Download:  moe-l2 model download --model <name> [--dir DIR]")
    return 0


def cmd_model_download(args):
    if args.model not in _MODEL_CATALOG:
        print(f"❌ Unknown model '{args.model}'. Available:")
        for name in _MODEL_CATALOG:
            print(f"  - {name}")
        return 1
    meta = _MODEL_CATALOG[args.model]
    out_dir = Path(args.dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"moe-l2 {__version__} — downloading {args.model} ({meta['size']})")
    print(f"  repo: {meta['repo']}")
    print(f"  → {out_dir}")
    ok = True
    for f in meta["files"]:
        url = _hf_url(meta["repo"], f)
        dest = out_dir / Path(f).name
        print(f"  [{f}]")
        if not _download_file(url, dest):
            ok = False
    if ok:
        print("✅ All files downloaded.")
        print(f"  Run: moe-l2 start --model {out_dir / Path(meta['files'][0]).name} --gpu")
        return 0
    print("⚠️ Some downloads failed — re-run to resume (partial files kept).")
    return 1


def cmd_start(args):
    print(f"moe-l2 {__version__} — starting L2 scheduler + proxy")

    # [moe-l2 2026-08-30 remote] 代理远程 llama-server（如 210 Windows 机器）：
    # 不启动本地二进制/不解析本地模型，proxy + gate（领域预测 + /moe-set-domain
    # 动态换表）照常工作。--model 传 model_id（默认 qwen）选 flywheel 表。
    remote_url = getattr(args, "backend_url", None)
    if remote_url:
        print("moe-l2 — remote proxy mode (no local llama-server)")
        print(f"  backend:  {remote_url}")
        backend_url = remote_url
        gate = None
        try:
            from .gate import RoutingProfiler
            from .predictor import load_mapping
            from .domain_router_flywheel import DomainRouterFlywheel
            from .router_table import model_id_from_path
            _model_arg = getattr(args, "model", None)
            _fw_model_id = (
                model_id_from_path(_model_arg)
                if _model_arg and _model_arg not in ("auto", "")
                else "qwen"
            )
            router_flywheel = DomainRouterFlywheel(model_id=_fw_model_id)
            gate = RoutingProfiler(
                cache=None,
                expert_map=load_mapping(model_id=_fw_model_id),
                router_flywheel=router_flywheel,
                router_server_url=backend_url,
            )
            print(f"  gate:     online routing adaptation ({backend_url}/moe-set-domain)")
            print(f"  flywheel: {_fw_model_id}")
        except Exception as e:
            logger.warning("Gate init failed (non-fatal): %s", e)
            gate = None
        print(f"  proxy:    127.0.0.1:{args.port} → {backend_url}")
        print()
        print("Connect your client to http://127.0.0.1:{}".format(args.port))
        print("Press Ctrl+C to stop")
        try:
            start_proxy(port=args.port, cache=None, backend_url=backend_url, gate=gate)
        except KeyboardInterrupt:
            pass
        return 0

    # Find model file
    model_path = _find_gguf(args.model)
    if model_path:
        print(f"  model:   {model_path}")
    else:
        print("  model:   (none found — using defaults)")
        model_path = None

    # Parse L2 size
    try:
        l2_size_str = args.l2_size
    except Exception:
        print(f"  invalid L2 size: {args.l2_size}")
        return 1

    # Initialize L2 cache
    cache_kwargs = {}
    if model_path:
        cache_kwargs["model_path"] = model_path

        # Compute slots from expert size directly
        from .gguf_reader import MoEGGUFReader

        # Multi-shard GGUF: the first shard (e.g. -00001-of-00003) usually
        # holds only metadata (no tensors). GGUFReader must point at a shard
        # that actually contains tensor data (verified 2026-08-05 with
        # DeepSeek-V4-Flash UD-IQ2_M: shard 1 = 5 MB metadata-only, 0 tensors).
        # Pick the largest sibling shard in the same directory.
        probe = Path(model_path)
        reader_path = model_path  # default: same file (single-shard case)
        if "-00001-of-" in probe.name:
            siblings = sorted(
                probe.parent.glob(probe.name.replace("-00001-of-", "-*-of-")),
                key=lambda p: p.stat().st_size,
                reverse=True,
            )
            if siblings:
                # llama-server MUST load from shard 1 (it auto-discovers the
                # sibling shards); only the metadata reader needs the data
                # shard. Keep both: model_path stays shard 1 for the server,
                # reader_path points at the largest shard for GGUF parsing.
                reader_path = str(siblings[0])
                cache_kwargs["model_path"] = reader_path
                print(f"  multishard: reading expert layout from {Path(reader_path).name}")

        reader = MoEGGUFReader(reader_path)
        expert_size = reader.per_expert_size()
        slots = _parse_l2_size(l2_size_str, expert_size)
        print(f"  expert:   {expert_size // (1024*1024)} MB each")
        print(f"  L2 size:  {l2_size_str} total → {slots} slots/layer")
        cache_kwargs["slots_per_layer"] = slots

    cache = L2Cache(**cache_kwargs)
    print(f"  layers:   {cache.n_layers}")
    print(f"  slots:    {cache.slots_per_layer}/layer = {cache.n_layers * cache.slots_per_layer} total")

    # ── GPU mode: spawn llama-server ──
    llama_proc: Optional[subprocess.Popen] = None
    backend_url = LLAMA_SERVER_BASE if args.gpu else "http://127.0.0.1:11434"

    if args.gpu:
        if not model_path:
            print("  ERROR: --gpu requires a model path (--model or auto-detect)")
            return 1
        print()
        print("  [GPU mode] Starting bundled llama-server (A3 tiered scheduling)...")

        # Ensure binaries are present
        if not _ensure_bins():
            return 1

        # Pre-flight: catch missing shared libs (e.g. libnccl.so.2) before
        # the 30s health poll, and give an actionable message.
        libs_ok, libs_detail = _check_dynamic_libs()
        if not libs_ok:
            print("  ERROR: llama-server dynamic libraries not satisfiable")
            print(f"    {libs_detail}")
            print("    Fix: run 'moe-l2 doctor' for details; if libnccl.so.2 is")
            print("    missing, update to a recent moe-l2 release (bins >= v0.2.1")
            print("    bundles libnccl) or install the NVIDIA nccl library.")
            return 1

        try:
            # [moe-l2 2026-08-13 路由表重构] 按模型名选表，缺失自动收集（用户不感知）。
            # 显式 --router-map 优先。
            router_map_path = getattr(args, "router_map", None)
            router_top_k = getattr(args, "router_top_k", 100)
            if router_map_path is None:
                from moe_l2.router_table import build_router_map_file

                data_dir = Path(__file__).resolve().parent.parent / "moe_l2" / "data"
                router_map_path = build_router_map_file(
                    model_path,
                    router_top_k=router_top_k,
                    data_dir=data_dir,
                    coverage_target=getattr(args, "coverage_target", 0.90),
                )
            llama_proc = _start_llama_server(
                model_path,
                _GPU_PORT,
                router_map_path,
                router_top_k,
                n_ctx=getattr(args, "ctx_size", None),
                n_parallel=getattr(args, "parallel", None),
            )
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            return 1

        # Wait for it to be ready
        print("  Waiting for llama-server to start...", end=" ", flush=True)
        ready = _wait_for_llama_server(_GPU_PORT)
        if not ready:
            print("TIMEOUT")
            stdout, stderr = llama_proc.communicate(timeout=5)
            print("  stdout:", stdout.decode(errors="replace")[-500:])
            print("  stderr:", stderr.decode(errors="replace")[-500:])
            llama_proc.kill()
            return 1
        print("READY")
        print(f"  backend:  127.0.0.1:{_GPU_PORT} (llama-server + CUDA + A3)")
        backend_url = f"http://127.0.0.1:{_GPU_PORT}"

    # ── 门控在线自适应（P2 ④）：实时路由 → 动态调缓存优先级 ──
    gate = None
    gate_thread = None
    router_flywheel = None
    if args.gpu and llama_proc is not None:
        from .gate import RoutingProfiler
        from .predictor import load_mapping
        try:
            # [moe-l2 2026-08-09] 路由表数据飞轮：真实路由按领域聚合 → 自动更新路由表
            # [2026-08-13 flywheel B] 按模型分文件：flywheel_{model_id}.json
            from .domain_router_flywheel import DomainRouterFlywheel
            from .router_table import model_id_from_path
            _fw_model_id = model_id_from_path(args.model)
            router_flywheel = DomainRouterFlywheel(model_id=_fw_model_id)
            gate = RoutingProfiler(
                cache=cache,
                expert_map=load_mapping(model_id=_fw_model_id),
                router_flywheel=router_flywheel,
                router_server_url=backend_url,
            )
            gate_thread = _spawn_gate_reader(llama_proc, gate)
            print("  gate:     online routing adaptation (LLAMA_EXPERT_LOG) enabled")
            print(f"  flywheel: router map auto-rebuild (domain_router_map_flywheel_{_fw_model_id}.json)")
        except Exception as e:
            logger.warning("Gate init failed (non-fatal): %s", e)
            gate = None

    # ── Start proxy ──
    print(f"  proxy:    127.0.0.1:{args.port} → {backend_url}")
    print()
    print("Connect your client to http://127.0.0.1:{}".format(args.port))
    print("Press Ctrl+C to stop")

    try:
        start_proxy(port=args.port, cache=cache, backend_url=backend_url, gate=gate)
    except KeyboardInterrupt:
        pass
    finally:
        if gate_thread is not None:
            gate_thread.stop()
        if llama_proc is not None:
            logger.info("Shutting down llama-server...")
            llama_proc.terminate()
            try:
                llama_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                llama_proc.kill()
                llama_proc.wait()
            logger.info("llama-server stopped")

    return 0


def cmd_stats(args):
    """Fetch and display cache stats from running proxy."""
    url = f"http://{DEFAULT_HOST}:{args.port}/stats"
    try:
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
    except httpx.ConnectError:
        print("moe-l2: proxy not running on 127.0.0.1:{}".format(args.port))
        print("  Start it first: moe-l2 start")
        return 1
    except Exception as e:
        print("moe-l2: failed to fetch stats:", e)
        return 1

    if "error" in data:
        print("moe-l2:", data["error"])
        return 1

    # Format the output nicely
    print("moe-l2 cache statistics")
    print(f"  Active domain:    {data.get('active_domain', 'N/A')}")
    print(f"  Requests:         {data.get('total_requests', 0):,}")
    print(f"  Hits:             {data.get('hits', 0):,}")
    print(f"  Misses:           {data.get('misses', 0):,}")
    print(f"  Hit rate:         {data.get('hit_rate_pct', 0)}%")
    print(f"  Utilization:      {data.get('utilization_pct', 0)}%")
    print(f"  Memory used:      {data.get('memory_usage_mb', 0)} MB")
    print(f"  Pinned experts:   {data.get('pinned_experts', 0):,}")
    print(f"  Pending loads:    {data.get('pending_loads', 0):,}")
    print(f"  Loads completed:  {data.get('loads_completed', 0):,}")

    per_layer = data.get("per_layer_lines", [])
    if per_layer:
        # Show first 3 + last 2 to avoid flooding
        shown = per_layer[:3]
        if len(per_layer) > 5:
            shown.append("  ...")
            shown.extend(per_layer[-2:])
        for line in shown:
            print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
