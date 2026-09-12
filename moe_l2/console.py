"""moe_l2/console.py — 控制台编码兜底（Windows GBK 控制台不再崩）

背景（2026-09-12 实机踩到）：中文 Windows 的控制台编码是 GBK（cp936），
`print` 里出现 GBK 无法表示的字符（⚠️ ✅ ❌ → 等）会抛 UnicodeEncodeError，
直接把整条命令崩掉——210 实测 `moe-l2 start` 在 router_table.py 打
"⚠️ 无路由表且无 llama-cli，跳过 selective pin" 时就崩了，服务起不来。
（日常在 210 上没暴露，是因为启动 bat 设了 PYTHONIOENCODING=utf-8。）

治本做法不是逐个把符号换成 ASCII（全仓库 180+ 处），而是在入口把
stdout/stderr 的 errors 改成 "replace"：控制台放不下字符时降级成 "?"，
而不是抛异常。编码不动（保持系统默认），所以中文照常显示。
"""

from __future__ import annotations

import sys


def enable_safe_console() -> None:
    """把 stdout/stderr 的编码错误策略改成 replace（幂等，可重复调用）。

    只在能改的时候改：非文本流（pythonw 下 stdout 可能是 None）、
    被重定向成非 TextIOWrapper 的对象等一律跳过，不影响正常路径。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)  # py3.7+ TextIOWrapper
        if reconfigure is None:
            # None / 非 TextIOWrapper（pythonw 下 stdout 可能是 None）→ 跳过
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):
            # 已关闭/不支持的流 → 保持原样，不影响正常路径
            continue
