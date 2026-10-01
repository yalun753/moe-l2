"""Tests for moe_l2.cli — pure helper functions only.

No subprocess / network / GPU code is exercised here; those paths are
covered by end-to-end verification (see references/ test reports).
"""

import pytest

from moe_l2 import cli

# ── _parse_l2_size ────────────────────────────────────────────────

class TestParseL2Size:
    def test_gb(self):
        assert cli._parse_l2_size("4GB", expert_size=1024) == 4 * 1024 * 1024

    def test_mb(self):
        assert cli._parse_l2_size("512MB", expert_size=1024 * 1024) == 512

    def test_lowercase(self):
        assert cli._parse_l2_size("2gb", expert_size=1024) == 2 * 1024 * 1024

    def test_whitespace(self):
        assert cli._parse_l2_size("  1GB  ", expert_size=1024) == 1024 * 1024

    def test_fractional(self):
        assert cli._parse_l2_size("1.5GB", expert_size=1024) == int(1.5 * 1024 * 1024)

    def test_min_one_slot(self):
        # A size smaller than one expert still reserves one slot
        assert cli._parse_l2_size("1MB", expert_size=10 * 1024 * 1024) == 1
        assert cli._parse_l2_size("0.5MB", expert_size=1024 * 1024) == 1

    def test_invalid(self):
        with pytest.raises(ValueError):
            cli._parse_l2_size("abc", expert_size=1024)
        with pytest.raises(ValueError):
            cli._parse_l2_size("", expert_size=1024)


# ── _hf_url ───────────────────────────────────────────────────────

class TestHfUrl:
    def test_basic(self):
        assert (
            cli._hf_url("unsloth/Qwen3.6-35B-A3B-GGUF", "model.gguf")
            == "https://hf-mirror.com/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/model.gguf"
        )


# ── _find_gguf ────────────────────────────────────────────────────

class TestFindGguf:
    def test_existing_hint(self, tmp_path):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"x")
        assert cli._find_gguf(str(f)) == str(f)

    def test_missing_hint_returns_none_or_path(self):
        # Falls back to /opt/data/models/*.gguf — may or may not exist.
        result = cli._find_gguf("/nonexistent/model.gguf")
        assert result is None or result.endswith(".gguf")


# ── doctor checks ─────────────────────────────────────────────────

class TestDoctorChecks:
    def test_check_python(self):
        ok, msg = cli._check_python()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_check_nvidia(self):
        ok, msg = cli._check_nvidia()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_check_cuda_lib(self):
        ok, msg = cli._check_cuda_lib()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_check_disk(self, tmp_path):
        ok, msg = cli._check_disk(tmp_path)
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_check_dynamic_libs(self):
        ok, msg = cli._check_dynamic_libs()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)


# ── [fix 2026-09-19 Windows] gate reader 不能依赖 select ──────────────

class _FakeGate:
    def __init__(self):
        self.lines: list[str] = []

    def on_log_line(self, line: str) -> None:
        self.lines.append(line)


def test_gate_reader_works_when_select_is_socket_only(monkeypatch, tmp_path):
    """gate reader 必须跟随日志文件工作，且不依赖 select。

    两层回归：① 引擎输出早就重定向到日志文件（proc.stderr 恒为 None），旧实现
    读 stderr 等于永不工作；② Windows 的 select 只支持 socket，对管道会抛
    WinError 10038 并被吞掉。任一条回归都会让飞轮聚合/漂移检测彻底失效
    （210 实测：日志 15 万行 EXPERT、飞轮记录 0、路由表永不写出）。
    """
    import select as _select
    import time as _time
    from moe_l2.cli import _GateReaderThread

    def _windows_like_select(*a, **kw):
        raise OSError(10038, "WinError 10038: not a socket")

    monkeypatch.setattr(_select, "select", _windows_like_select)

    log = tmp_path / "llama-server.log"
    log.write_text("", encoding="utf-8")
    gate = _FakeGate()
    th = _GateReaderThread(None, gate, log_path=log)
    # 竞态：reader 线程启动时 seek 到文件末尾——若 append 发生在那之前，这一行会被当成
    # "历史"跳过（读不到就永久等不到）。所以反复追加，每次追加都晚于 seek，直到读进来。
    for _ in range(50):
        with open(log, "a", encoding="utf-8") as f:
            f.write("EXPERT|L0|T1: [1,2,3,4,5,6,7,8]\n")
        for _ in range(10):
            if gate.lines:
                break
            _time.sleep(0.02)
        if gate.lines:
            break
    th.stop()
    assert gate.lines, "gate reader 没喂进任何日志行"
    assert gate.lines[0].startswith("EXPERT|L0|T1:")


def test_gate_reader_skips_history_and_reads_new_lines(tmp_path):
    """只读启动之后的新行：历史几十万行不能被重放（否则飞轮会爆量重建）。"""
    import time as _time
    from moe_l2.cli import _GateReaderThread

    log = tmp_path / "llama-server.log"
    log.write_text("EXPERT|L0|T1: [9,9,9,9,9,9,9,9]\n", encoding="utf-8")
    gate = _FakeGate()
    th = _GateReaderThread(None, gate, log_path=log)
    _time.sleep(0.3)
    assert gate.lines == []          # 历史行被跳过
    with open(log, "a", encoding="utf-8") as f:
        f.write("EXPERT|L0|T1: [1,2,3,4,5,6,7,8]\n")
    for _ in range(150):
        if gate.lines:
            break
        _time.sleep(0.02)
    th.stop()
    assert gate.lines and gate.lines[0].startswith("EXPERT|L0|T1: [1,")
