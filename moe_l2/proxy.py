"""Ollama transparent proxy — intercepts /api/generate requests.

Sits between the user and ollama, adding L2 cache preloading
based on domain prediction before forwarding the request.
"""

import json
import logging
import mmap
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import httpx

from .cache import L2Cache
from .predictor import enable_tfidf, load_mapping, predict_hybrid
from .training_flywheel import append_sample, maybe_retrain, training_stats

if False:  # TYPE_CHECKING only
    from .gate import RoutingProfiler  # noqa: F401

logger = logging.getLogger("moe-l2-proxy")

# [moe-l2 2026-09-05] 监听地址支持 env 覆盖：MOE_L2_HOST=0.0.0.0 可让局域网
# 其他设备（如 210 WebUI / Hermes）经本 proxy 走全链路（L0 领域预测 + 自动换表）。
DEFAULT_HOST = os.environ.get("MOE_L2_HOST", "127.0.0.1")
DEFAULT_PORT = 11435
OLLAMA_BASE = "http://127.0.0.1:11434"
LLAMA_SERVER_BASE = "http://127.0.0.1:11436"

# L0a × 动态 pin 联动（2026-08-09，见 历史记录文档/方案-L0a动态pin联动-20260809.md）
# MOE_L2_MODEL_PATH: 模型 GGUF 路径（预 touch 需要）；不设 = 跳过
# MOE_L2_PRETOUCH_GB: 预 touch 预算（默认 8GB，0 = 关闭）
_PRETOUCH_GB = float(os.environ.get("MOE_L2_PRETOUCH_GB", "8"))
_MODEL_PATH = os.environ.get("MOE_L2_MODEL_PATH", "") or None
_PAGE = 4096


class _ExpertPretoucher:
    """把预测领域的专家页提前喂进 page cache。

    动态 pin（C++）首次注册专家时对 mmap 页做 cudaHostRegister，缺页要付
    ~1.7ms 读盘。这里用预测领域 → 专家 ID → GGUF 文件偏移，在请求间隙
    后台 touch 这些页（每页读 1 字节），让首次 fault 命中 page cache 免读盘。
    """

    def __init__(self, model_path: str, budget_gb: float):
        self.model_path = model_path
        self.budget_bytes = int(budget_gb * 1024 ** 3)
        self._shard_readers: list[tuple[str, object]] = []
        self._tensor_index: dict[str, tuple[str, object]] = {}
        self._pages_cache: dict[str, list[tuple[str, int]]] = {}
        self._lock = threading.Lock()
        self._EXPERT_PATTERNS = [
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ]

    def _load_reader(self):
        """多分片 GGUF 索引：主片可能只有元数据（0 tensor），
        权重 tensor 分布在 -0000X-of-0000Y.gguf 分片里。
        field.offset 是分片内偏移，touch 时按分片文件分别 open。"""
        if self._shard_readers:
            return
        import glob

        from gguf import GGUFReader
        d = os.path.dirname(os.path.abspath(self.model_path))
        b = os.path.basename(self.model_path)
        pattern = b.replace("-00001-of-", "-*-of-") if "-00001-of-" in b else b
        shards = sorted(glob.glob(os.path.join(d, pattern)))
        if not shards:
            shards = [self.model_path]
        for sp in shards:
            rd = GGUFReader(sp)
            self._shard_readers.append((sp, rd))
            for t in rd.tensors:
                self._tensor_index[t.name] = (sp, t)
        logger.info(
            "Pretouch index: %d shards, %d tensors",
            len(shards), len(self._tensor_index))

    def _expert_pages_for(self, domain: str, expert_map: dict) -> list[tuple[str, int]]:
        """预测领域专家 → (分片路径, 页偏移) 列表（去重、预算截断）。"""
        with self._lock:
            if domain in self._pages_cache:
                return self._pages_cache[domain]
            self._load_reader()
            domains = (expert_map or {}).get("domains", {})
            per_layer = (domains.get(domain, {}) or {}).get(
                "per_layer_domain_preferred", {})
            pages: list[tuple[str, int]] = []
            seen: set[int] = set()
            spent = 0
            for layer_str, expert_ids in per_layer.items():
                try:
                    layer = int(layer_str)
                except (TypeError, ValueError):
                    continue
                for eid in expert_ids or []:
                    for pattern in self._EXPERT_PATTERNS:
                        hit = self._tensor_index.get(f"blk.{layer}.{pattern}")
                        if hit is None:
                            continue  # 分片/层不匹配，跳过
                        shard_path, t = hit
                        bpe = t.n_bytes // t.data.shape[0]
                        start = t.field.offset + int(eid) * bpe
                        end = min(start + bpe, t.field.offset + t.n_bytes)
                        pg = start // _PAGE * _PAGE
                        while pg < end and spent < self.budget_bytes:
                            if pg not in seen:
                                seen.add(pg)
                                pages.append((shard_path, pg))
                                spent += _PAGE
                            pg += _PAGE
                        if spent >= self.budget_bytes:
                            break
                    if spent >= self.budget_bytes:
                        break
                if spent >= self.budget_bytes:
                    break
            self._pages_cache[domain] = pages
            logger.info(
                "Pretouch offsets domain=%s: %d pages (%.2f GB)",
                domain, len(pages), spent / 1024 ** 3)
            return pages

    def touch(self, domain: str, expert_map: dict):
        """后台线程入口：touch 预测领域专家页。失败只警告，不影响转发。"""
        if not self.model_path or not os.path.exists(self.model_path):
            return
        try:
            pages = self._expert_pages_for(domain, expert_map)
            if not pages:
                return
            # 按分片分组，逐文件 mmap touch
            by_shard: dict[str, list[int]] = {}
            for sp, pg in pages:
                by_shard.setdefault(sp, []).append(pg)
            for sp, pgs in by_shard.items():
                with open(sp, "rb") as f:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    try:
                        for pg in pgs:
                            _ = mm[pg]  # 每页读 1 字节 → 页驻留 page cache
                    finally:
                        mm.close()
            logger.info(
                "Pretouched %d pages for domain=%s (%d shards)",
                len(pages), domain, len(by_shard))
        except Exception as e:
            logger.warning("Pretouch failed (non-fatal): %s", e)


# Headers NOT forwarded to the client (hop-by-hop or internal)
_HOP_BY_HOP = frozenset({
    "content-length", "content-encoding", "transfer-encoding",
    "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailers", "upgrade", "server",
})


class MoEL2ProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler that proxies to backend with L2 preloading."""

    cache: Optional[L2Cache] = None
    backend_url: str = OLLAMA_BASE  # overridden per-instance by start_proxy
    gate: Optional["RoutingProfiler"] = None  # online gate adaptation (optional)

    # ── Domain prediction & preloading ──────────────────────────

    def _predict_and_preload(self, text: str):
        """Predict domain and trigger expert preload + flywheel sampling."""
        # L0a touch 独立于 L2 cache：动态 pin 场景不依赖 /dev/shm
        touch_enabled = bool(_MODEL_PATH and _PRETOUCH_GB > 0)
        if not self.cache and not touch_enabled:
            # [2026-08-29 210 自动换表] cache 未启用时仍要让 gate 换表：
            # proxy 转发远程 backend（如 210），gate.on_request 负责预测领域
            # → POST /moe-set-domain（领域切换 = 热区刷新）。放行纯 gate 模式。
            try:
                # 三层预测：关键词 → TF-IDF 分类器 → 语义兜底
                enable_tfidf()
                domain = predict_hybrid(text)
                logger.info("Predicted domain: %s", domain)
                if getattr(self, "gate", None) is not None:
                    try:
                        self.gate.on_request(domain)
                    except Exception as ge:
                        logger.warning("Gate request signal failed (non-fatal): %s", ge)
            except Exception:
                pass
            return
        try:
            # 三层预测：关键词 → TF-IDF 分类器 → 语义兜底（提升样本标签质量）
            enable_tfidf()  # 惰性加载，缺 sklearn 时静默降级
            domain = predict_hybrid(text)
            logger.info("Predicted domain: %s", domain)

            # L0a × 动态 pin 联动：后台预 touch 预测领域专家页（不阻塞转发）
            if touch_enabled:
                try:
                    if getattr(self, "_pretoucher", None) is None:
                        self._pretoucher = _ExpertPretoucher(
                            _MODEL_PATH, _PRETOUCH_GB)
                    from .router_table import model_id_from_path
                    expert_map = load_mapping(
                        model_id=model_id_from_path(_MODEL_PATH) if _MODEL_PATH else None)
                    t = threading.Thread(
                        target=self._pretoucher.touch,
                        args=(domain, expert_map), daemon=True)
                    t.start()
                except Exception as pe:
                    logger.warning("Pretouch thread failed (non-fatal): %s", pe)

            if not self.cache:
                return
            from .router_table import model_id_from_path
            expert_map = load_mapping(
                model_id=model_id_from_path(_MODEL_PATH) if _MODEL_PATH else None)
            self.cache.preload_domain(domain, expert_map)
            logger.info("Preloaded experts for domain=%s", domain)

            # 门控在线自适应（P2 ④）：请求级信号 → 预热目标域
            if getattr(self, "gate", None) is not None:
                try:
                    self.gate.on_request(domain)
                except Exception as ge:
                    logger.warning("Gate request signal failed (non-fatal): %s", ge)

            # 模式 B 数据飞轮：记录真实流量样本，攒够阈值增量重训
            # [moe-l2 2026-08-19] maybe_retrain 放后台线程：旧版同步全量重训
            # （TF-IDF + LinearSVC，数百样本可达数秒）在并发请求下阻塞请求
            # 线程 → proxy 并发卡死（与换表同步阻塞同族）。
            try:
                append_sample(text, domain)
                threading.Thread(target=maybe_retrain, daemon=True,
                                 name="moe-l2-flywheel-retrain").start()
            except Exception as fe:
                logger.warning("Flywheel sampling failed (non-fatal): %s", fe)
        except Exception as e:
            # Non-fatal: forwarding still happens even if preload fails
            logger.warning("Preload failed: %s", e)

    # ── Entry points ────────────────────────────────────────────

    def do_GET(self):
        if self.path == "/stats":
            self._handle_stats()
        elif self.path == "/health":
            self._send_json({"status": "ok", "cache_active": self.cache is not None})
        else:
            self._proxy_request("GET")

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        if self.path == "/api/generate":
            self._handle_api("generate", body)
        elif self.path == "/api/chat":
            self._handle_api("chat", body)
        elif self.path.startswith("/v1/"):
            # OpenAI API format: pass through, but still do prediction
            try:
                data = json.loads(body)
                if "messages" in data:
                    text = data["messages"][-1].get("content", "")
                elif "prompt" in data:
                    text = data["prompt"]
                else:
                    text = ""
                if text:
                    self._predict_and_preload(text)
            except Exception:
                pass
            self._proxy_request("POST", body)
        else:
            self._proxy_request("POST", body)

    # ── API handlers ────────────────────────────────────────────

    def _handle_api(self, endpoint: str, body: bytes):
        """Handle /api/generate or /api/chat — predict, preload, forward.

        Backend endpoint mapping:
          - ollama (11434):  /api/generate → /api/generate, /api/chat → /api/chat
          - llama-server (11436): /api/generate → /completion,
                                  /api/chat → /v1/chat/completions (OpenAI format)
        """
        try:
            data = json.loads(body)
            if endpoint == "generate":
                text = data.get("prompt", "")
            else:
                messages = data.get("messages", [])
                text = messages[-1].get("content", "") if messages else ""
            if text:
                self._predict_and_preload(text)
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logger.warning("Failed to parse request body: %s", e)

        # 后端端点映射：llama-server 用 OpenAI 兼容端点，不是 ollama /api/*
        if endpoint == "chat" and "11436" in self.backend_url:
            self._forward_to_backend(body, "/v1/chat/completions")
        elif endpoint == "generate" and "11436" in self.backend_url:
            self._forward_to_backend(body, "/completion")
        else:
            self._forward_to_backend(body, f"/api/{endpoint}")

    # ── Backend forwarding ───────────────────────────────────────

    def _forward_to_backend(self, body: bytes, path: str):
        """Forward request to backend, handling streaming or blocking."""
        url = f"{self.backend_url}{path}"

        try:
            data = json.loads(body)
            is_stream = data.get("stream", True)
        except (json.JSONDecodeError, KeyError):
            is_stream = False

        client = httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0))
        try:
            if is_stream:
                self._forward_stream(client, url, body)
            else:
                self._forward_blocking(client, url, body)
        finally:
            client.close()

    def _forward_stream(self, client: httpx.Client, url: str, body: bytes):
        """SSE-stream the response from backend to the client."""
        try:
            with client.stream(
                "POST",
                url,
                content=body,
                headers={"Content-Type": "application/json"},
            ) as resp:
                self.send_response(resp.status_code)
                # SSE-specific headers
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                # HTTP/1.0 服务器无 Content-Length/chunked，客户端只能靠连接
                # 关闭判断流结束——不能声明 keep-alive，否则流式响应挂死。
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                for chunk in resp.iter_bytes():
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()
        except httpx.ConnectError:
            self._send_error(502, f"cannot connect to backend at {self.backend_url}")
        except httpx.TimeoutException:
            self._send_error(504, "backend request timed out")

    def _forward_blocking(self, client: httpx.Client, url: str, body: bytes):
        """Non-streaming forward: wait for full response, return as-is."""
        try:
            resp = client.post(
                url,
                content=body,
                headers={"Content-Type": "application/json"},
            )
            self.send_response(resp.status_code)
            for key, value in resp.headers.items():
                if key.lower() not in _HOP_BY_HOP:
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(resp.content)
        except httpx.ConnectError:
            self._send_error(502, f"cannot connect to backend at {self.backend_url}")
        except httpx.TimeoutException:
            self._send_error(504, "backend request timed out")

    def _proxy_request(self, method: str, body: bytes = None):
        """Generic HTTP proxy for endpoints like /api/tags, /api/show, etc."""
        url = f"{self.backend_url}{self.path}"
        client = httpx.Client(timeout=600.0)
        try:
            req = client.build_request(method, url, content=body)
            resp = client.send(req)
            self.send_response(resp.status_code)
            for key, value in resp.headers.items():
                if key.lower() not in _HOP_BY_HOP:
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(resp.content)
        except httpx.ConnectError:
            self._send_error(502, f"cannot connect to backend at {self.backend_url}")
        finally:
            client.close()

    def _send_json(self, data: dict, status: int = 200):
        """Send a JSON response."""
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str):
        """Send a JSON error response."""
        logger.error(message)
        self._send_json({"error": message}, status)

    def _handle_stats(self):
        """Return cache statistics as JSON (for 'moe-l2 stats')."""
        if not self.cache:
            self._send_json({"error": "cache not initialized"}, status=503)
            return
        try:
            stats = self.cache.stats()
            # 模式 B 数据飞轮状态
            stats.update(training_stats())
            # Return compact version for CLI consumption
            self._send_json(stats)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def log_message(self, fmt, *args):
        logger.info(fmt, *args)


def start_proxy(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    cache: Optional[L2Cache] = None,
    backend_url: str = OLLAMA_BASE,
    gate: Optional["RoutingProfiler"] = None,
):
    """Start the moe-l2 proxy server.

    Args:
        host: Listen address.
        port: Listen port.
        cache: Optional L2Cache instance for domain preloading.
        backend_url: Upstream inference server URL.
        gate: Optional RoutingProfiler for online gate adaptation.
    """
    MoEL2ProxyHandler.cache = cache
    MoEL2ProxyHandler.gate = gate
    # Patch the backend URL onto the handler
    MoEL2ProxyHandler.backend_url = backend_url
    server = HTTPServer((host, port), MoEL2ProxyHandler)
    server.timeout = 0.5  # allow KeyboardInterrupt to work
    logger.info("moe-l2 proxy listening on %s:%s → %s", host, port, backend_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.server_close()
