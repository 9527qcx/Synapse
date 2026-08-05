"""日志系统：统一格式 + trace_id 关联 + Windows 中文编码修复。

用法（程序入口处调用一次）：
    init_logging()
    logger = logging.getLogger(__name__)   # 各模块用 logging.getLogger 获取
"""
from __future__ import annotations

import logging
import sys

from .tracing import get_trace_id

_LOGGING_INITIALIZED = False


class _TraceIdFormatter(logging.Formatter):
    """把当前上下文的 trace_id 注入每行日志。"""

    def format(self, record: logging.LogRecord) -> str:
        record.trace_id = get_trace_id() or "-"
        return super().format(record)


def init_logging(level: int = logging.INFO) -> None:
    """初始化根日志器（幂等：重复调用无副作用）。"""
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return
    _LOGGING_INITIALIZED = True

    # Windows 控制台中文乱码修复：stdout 与 stderr 都要强制 UTF-8
    # （logging 的 StreamHandler 默认写 stderr，只修 stdout 不够）
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    handler = logging.StreamHandler()
    handler.setFormatter(
        _TraceIdFormatter(
            "%(asctime)s [%(levelname)s] [%(name)s] [trace=%(trace_id)s] %(message)s"
        )
    )
    root = logging.getLogger()
    # 避免重复添加（幂等保护的二道保险）
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    root.setLevel(level)
