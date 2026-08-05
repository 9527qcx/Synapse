"""observability 单元测试：no-op 降级、trace_id 贯穿、守卫行为。"""
from __future__ import annotations

import pytest

from synapselib.config.settings import Settings
from synapselib.observability import langfuse_setup, tracing


class TestTraceSpanNoop:
    """tracer 为 None（默认状态）时：不抛异常、trace_id 正常维护。"""

    def teardown_method(self):
        tracing.set_tracer(None)

    def test_noop_does_not_raise(self):
        with tracing.trace_span("smoke", foo="bar"):
            pass

    def test_trace_id_generated_inside(self):
        with tracing.trace_span("span1"):
            tid = tracing.get_trace_id()
        assert tid  # 非空
        assert tracing.get_trace_id() == ""  # 退出后上下文恢复

    def test_nested_spans_share_trace_id(self):
        with tracing.trace_span("outer"):
            outer_id = tracing.get_trace_id()
            with tracing.trace_span("inner"):
                assert tracing.get_trace_id() == outer_id

    def test_exception_propagates(self):
        with pytest.raises(ValueError, match="业务异常"):
            with tracing.trace_span("boom"):
                raise ValueError("业务异常")
        assert tracing.get_trace_id() == ""

    def test_get_trace_url_none_without_tracer(self):
        with tracing.trace_span("x"):
            assert tracing.get_trace_url() is None

    def test_traced_decorator(self):
        @tracing.traced("decorated_fn")
        def add(a: int, b: int) -> int:
            return a + b

        assert add(1, 2) == 3


class _FakeSpan:
    def __init__(self):
        self.input = None
        self.output = None
        self.ended = False

    def update(self, **kwargs):
        if "input" in kwargs:
            self.input = kwargs["input"]
        if "output" in kwargs:
            self.output = kwargs["output"]

    def end(self):
        self.ended = True


class _FakeTracer:
    """模拟 langfuse 客户端的最小接口。"""

    def __init__(self):
        self.spans: list[_FakeSpan] = []
        self.events: list[str] = []

    def span(self, name: str, span_type: str):
        s = _FakeSpan()
        self.spans.append(s)
        return s

    def event(self, name: str, **kwargs):
        self.events.append(name)


class TestTraceSpanWithTracer:
    def teardown_method(self):
        tracing.set_tracer(None)

    def test_span_recorded_and_ended(self):
        fake = _FakeTracer()
        tracing.set_tracer(fake, host="https://example.com")
        with tracing.trace_span("检索", span_type="tool", q="幻觉"):
            pass
        assert len(fake.spans) == 1
        assert fake.spans[0].ended is True
        assert fake.spans[0].input == {"q": "幻觉"}
        assert "latency_ms" in fake.spans[0].output

    def test_error_recorded_and_reraiset(self):
        fake = _FakeTracer()
        tracing.set_tracer(fake)
        try:
            with tracing.trace_span("boom"):
                raise RuntimeError("炸了")
        except RuntimeError:
            pass
        assert fake.spans[0].output["error"] == "炸了"

    def test_event_recorded(self):
        fake = _FakeTracer()
        tracing.set_tracer(fake)
        tracing.record_event("task_completed", task_id="t1")
        assert fake.events == ["task_completed"]

    def test_trace_url_built(self):
        fake = _FakeTracer()
        tracing.set_tracer(fake, host="https://cloud.langfuse.com")
        with tracing.trace_span("x"):
            tid = tracing.get_trace_id()
            url = tracing.get_trace_url()
        assert url is not None
        assert url == f"https://cloud.langfuse.com/trace/{tid}"


class TestInitLangfuse:
    def teardown_method(self):
        tracing.set_tracer(None)

    def test_disabled_without_keys(self):
        s = Settings(langfuse_public_key="", langfuse_secret_key="")
        langfuse_setup.init_langfuse(s)
        assert tracing._tracer is None  # noqa: SLF001 直接断言内部状态

    def test_with_fake_keys_does_not_raise(self):
        s = Settings(langfuse_public_key="pk-fake", langfuse_secret_key="sk-fake")
        langfuse_setup.init_langfuse(s)
        # 无论初始化成功与否，业务侧都不应受影响（tracer 可能是 client 或 None）
        with tracing.trace_span("smoke"):
            pass
