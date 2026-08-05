"""追踪基础设施：trace_span / traced / record_event。

设计原则（大纲 §13「阶段 1 即建立完整日志体系」）：
- 不直接使用 langfuse 的 @observe 装饰器（与全局 client 强耦合，无法守卫降级）
- 本模块是唯一与追踪相关的入口，内部对 langfuse 的调用全部 try/except 包裹：
  追踪失败只打 warning，绝不抛出，绝不影响业务
- trace_id 通过 contextvars 贯穿：日志行与 LangFuse 面板可通过 trace_id 互查
"""
from __future__ import annotations

import functools
import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# trace_id 上下文变量（logging_config.py 的 formatter 从这里读取）
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")

# langfuse 客户端（None = no-op 模式）；host 用于拼 trace URL
_tracer: Any = None
_host: str | None = None


def set_tracer(tracer: Any, host: str | None = None) -> None:
    """注入追踪后端（由 langfuse_setup.init_langfuse 调用）。"""
    global _tracer, _host
    _tracer = tracer
    _host = host


def get_trace_id() -> str:
    """当前上下文的 trace_id（无则空串）。"""
    return _trace_id_var.get()


@contextmanager
def trace_span(name: str, *, span_type: str = "general", **attrs: Any) -> Iterator[None]:
    """记录一个追踪区间（函数体/代码块）。

    - tracer 未启用时仍维护 trace_id，保证日志关联不丢
    - 异常时 span 记录 error 并重新抛出，调用方原有异常语义不变
    """
    trace_id = _trace_id_var.get() or uuid.uuid4().hex
    token = _trace_id_var.set(trace_id)
    span = None
    start = time.perf_counter()

    if _tracer is not None:
        try:
            span = _tracer.span(name=name, span_type=span_type)
            if attrs:
                span.update(input=attrs)
        except Exception as e:  # noqa: BLE001 守卫：追踪失败不影响业务
            logger.warning("trace_span 开启失败 [%s]: %s", name, e)
            span = None

    try:
        yield
    except Exception as e:
        if span is not None:
            try:
                span.update(output={"error": str(e)}, level="ERROR")
            except Exception:  # noqa: BLE001
                pass
        raise
    else:
        if span is not None:
            try:
                span.update(output={"latency_ms": round((time.perf_counter() - start) * 1000, 2)})
            except Exception:  # noqa: BLE001
                pass
    finally:
        if span is not None:
            try:
                span.end()
            except Exception:  # noqa: BLE001
                pass
        _trace_id_var.reset(token)


def traced(name: str, **attrs: Any):
    """trace_span 的函数装饰器版。"""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            with trace_span(name, **attrs):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def record_event(name: str, **attrs: Any) -> None:
    """轻量事件点（如 'task_completed'、'dedup_skipped'）。no-op 模式直接跳过。"""
    if _tracer is None:
        return
    try:
        _tracer.event(name=name, input=attrs or None)
    except Exception as e:  # noqa: BLE001
        logger.warning("record_event 失败 [%s]: %s", name, e)


def get_trace_url() -> str | None:
    """当前追踪的 LangFuse 面板 URL（未启用追踪时返回 None）。"""
    trace_id = _trace_id_var.get()
    if not trace_id or _tracer is None or not _host:
        return None
    return f"{_host}/trace/{trace_id}"
