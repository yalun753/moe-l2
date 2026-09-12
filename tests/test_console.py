"""test_console.py — 控制台编码兜底（Windows GBK 控制台不再因 print 符号崩掉）

背景：中文 Windows 控制台是 GBK，print(⚠️/✅/❌) 会抛 UnicodeEncodeError 崩掉整条
命令（210 实测：moe-l2 start 在 router_table 打警告时直接挂，服务起不来）。
"""
import io
import sys

import pytest

from moe_l2.console import enable_safe_console


def _gbk_stream():
    """GBK 严格模式的 stdout（= 中文 Windows 控制台等价物）。"""
    return io.TextIOWrapper(io.BytesIO(), encoding="gbk", errors="strict")


def test_gbk_console_crashes_without_fix(monkeypatch):
    """先证明这就是崩溃原因：GBK 控制台打 ⚠️ 必抛 UnicodeEncodeError。"""
    stream = _gbk_stream()
    monkeypatch.setattr(sys, "stdout", stream)

    with pytest.raises(UnicodeEncodeError):
        print("  [router] ⚠️ 无路由表且无 llama-cli，跳过 selective pin")


def test_enable_safe_console_prevents_crash(monkeypatch):
    """enable_safe_console() 之后同样一行不再崩，符号降级成 '?'。"""
    stream = _gbk_stream()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", _gbk_stream())

    enable_safe_console()
    print("  [router] ⚠️ 无路由表且无 llama-cli，跳过 selective pin")
    stream.flush()

    out = stream.buffer.getvalue().decode("gbk", errors="replace")
    assert "无路由表" in out  # 中文照常（编码没动）
    assert "⚠" not in out  # 放不下的符号被替换掉，而不是抛异常


def test_enable_safe_console_is_safe_on_odd_streams(monkeypatch):
    """stdout/stderr 是 None 或没有 reconfigure 时不能炸（pythonw / 被重定向）。"""
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", object())

    enable_safe_console()  # 不抛即通过
    enable_safe_console()  # 幂等，可重复调用


def test_cli_main_enables_safe_console(monkeypatch):
    """CLI 入口（moe-l2 start 等所有子命令的必经之路）必须先装兜底。"""
    import moe_l2.cli as cli
    import moe_l2.console as console

    called = []
    monkeypatch.setattr(console, "enable_safe_console", lambda: called.append(True))
    monkeypatch.setattr(sys, "argv", ["moe-l2", "--version"])

    with pytest.raises(SystemExit):
        cli.main()  # argparse 的 --version 会 SystemExit(0)

    assert called == [True]
