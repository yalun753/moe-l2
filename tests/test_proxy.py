"""Tests for moe_l2.proxy — end-to-end HTTP forwarding against a fake backend.

Starts a real MoEL2ProxyHandler server (via start_proxy) plus a tiny
fake inference backend on localhost, then exercises the HTTP surface:
health, stats, ollama /api/generate (blocking + SSE), /api/chat and
OpenAI-style /v1/ pass-through. No GPU / real model needed.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from moe_l2.cache import L2Cache
from moe_l2.proxy import start_proxy


class FakeBackend(BaseHTTPRequestHandler):
    """Minimal fake inference server with blocking + SSE responses."""

    def _respond_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b'data: {"response":"tok1"}\n\n')
        self.wfile.write(b'data: {"response":"tok2"}\n\n')
        self.wfile.flush()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}
        if data.get("stream"):
            self._respond_sse()
        else:
            self._respond_json({"response": "hello", "done": True})

    def do_GET(self):
        self._respond_json({"models": ["fake-model"]})

    def log_message(self, format, *args):
        pass


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def cache(tmp_path):
    c = L2Cache(
        n_layers=2,
        slots_per_layer=4,
        expert_size=64,
        l2_dir=tmp_path / "l2",
    )
    yield c
    c.close()


@pytest.fixture
def backend():
    server = HTTPServer(("127.0.0.1", 0), FakeBackend)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture
def proxy(backend, cache, tmp_path, monkeypatch):
    """Start the moe-l2 proxy in a daemon thread; return its base URL."""
    # Never touch the real user's flywheel sample file during tests
    monkeypatch.setattr("moe_l2.proxy.append_sample", lambda *a, **k: 0)
    monkeypatch.setattr("moe_l2.proxy.maybe_retrain", lambda *a, **k: None)

    port = _free_port()
    t = threading.Thread(
        target=start_proxy,
        kwargs=dict(
            host="127.0.0.1",
            port=port,
            cache=cache,
            backend_url=backend,
        ),
        daemon=True,
    )
    t.start()
    base = f"http://127.0.0.1:{port}"

    # Wait for the server to accept connections
    for _ in range(50):
        try:
            httpx.get(f"{base}/health", timeout=0.5)
            break
        except httpx.HTTPError:
            threading.Event().wait(0.1)
    return base


# ── Health / stats ────────────────────────────────────────────────

class TestHealthStats:
    def test_health(self, proxy):
        r = httpx.get(f"{proxy}/health", timeout=5)
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "cache_active": True}

    def test_stats(self, proxy):
        r = httpx.get(f"{proxy}/stats", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "hits" in data
        assert "total_slots" in data
        assert "active_domain" in data


# ── Request forwarding ────────────────────────────────────────────

class TestForwarding:
    def test_generate_blocking(self, proxy):
        r = httpx.post(
            f"{proxy}/api/generate",
            json={"prompt": "hello", "stream": False},
            timeout=10,
        )
        assert r.status_code == 200
        assert r.json()["response"] == "hello"

    def test_generate_stream_sse(self, proxy):
        with httpx.stream(
            "POST",
            f"{proxy}/api/generate",
            json={"prompt": "hello", "stream": True},
            timeout=10,
        ) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            body = "".join(chunk.decode() for chunk in r.iter_bytes())
        assert 'data: {"response":"tok1"}' in body
        assert 'data: {"response":"tok2"}' in body

    def test_chat_blocking(self, proxy):
        r = httpx.post(
            f"{proxy}/api/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
            timeout=10,
        )
        assert r.status_code == 200
        assert r.json()["response"] == "hello"

    def test_openai_v1_passthrough(self, proxy):
        r = httpx.post(
            f"{proxy}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
            timeout=10,
        )
        assert r.status_code == 200

    def test_generic_get_proxied(self, proxy):
        r = httpx.get(f"{proxy}/api/tags", timeout=5)
        assert r.status_code == 200
        assert r.json()["models"] == ["fake-model"]


# ── Domain preload side effect ────────────────────────────────────

class TestPreloadSideEffect:
    def test_generate_triggers_domain_prediction(self, proxy, cache):
        # "implement a sorting algorithm" → codegen keyword hit
        r = httpx.post(
            f"{proxy}/api/generate",
            json={"prompt": "implement a sorting algorithm", "stream": False},
            timeout=10,
        )
        assert r.status_code == 200
        assert cache.stats()["active_domain"] == "codegen"

    def test_openai_passthrough_triggers_prediction(self, proxy, cache):
        r = httpx.post(
            f"{proxy}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "翻译这段话"}], "stream": False},
            timeout=10,
        )
        assert r.status_code == 200
        assert cache.stats()["active_domain"] == "translate"


# ── No-cache mode ─────────────────────────────────────────────────

class TestNoCache:
    def test_stats_503_without_cache(self, backend, tmp_path):
        port = _free_port()
        t = threading.Thread(
            target=start_proxy,
            kwargs=dict(host="127.0.0.1", port=port, cache=None, backend_url=backend),
            daemon=True,
        )
        t.start()
        base = f"http://127.0.0.1:{port}"
        r = None
        for _ in range(50):
            try:
                r = httpx.get(f"{base}/health", timeout=0.5)
                break
            except httpx.HTTPError:
                threading.Event().wait(0.1)
        assert r is not None and r.json()["cache_active"] is False
        r = httpx.get(f"{base}/stats", timeout=5)
        assert r.status_code == 503


# ── [sagasm C1 2026-09-19] /v1/models 短名改写 ─────────────────────

def test_rewrite_model_ids_with_alias():
    from moe_l2.proxy import rewrite_model_ids

    payload = {"object": "list", "data": [{"id": "D:/models/qwen.gguf"}, {"id": "x"}]}
    out = rewrite_model_ids(payload, "qwen3-35b")
    assert [m["id"] for m in out["data"]] == ["qwen3-35b", "qwen3-35b"]


def test_rewrite_model_ids_without_alias_passthrough():
    from moe_l2.proxy import rewrite_model_ids

    payload = {"data": [{"id": "D:/models/qwen.gguf"}]}
    assert rewrite_model_ids(payload, None)["data"][0]["id"] == "D:/models/qwen.gguf"
    assert rewrite_model_ids(payload, "")["data"][0]["id"] == "D:/models/qwen.gguf"


def test_rewrite_model_ids_tolerates_weird_payload():
    from moe_l2.proxy import rewrite_model_ids

    assert rewrite_model_ids(None, "a") is None
    assert rewrite_model_ids({"data": "not-a-list"}, "a") == {"data": "not-a-list"}
    assert rewrite_model_ids({"nope": 1}, "a") == {"nope": 1}
