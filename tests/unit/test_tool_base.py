"""tools/base.py 单元测试：重试语义 + ToolBox。

重试是本里程碑的**核心知识点**，测透四条语义：
1. 失败后重试能成功 → 返回结果
2. 全部失败 → RetryExhaustedError（继承 ToolError，调用方按工具失败处理）
3. 非 ToolError（如 ValueError）→ 不重试，直接穿透
4. 退避延迟翻倍：0.5s → 1.0s → 2.0s……
"""
from __future__ import annotations

import pytest

from synapselib.config.settings import Settings
from synapselib.core.errors import RetryExhaustedError, ToolError
from synapselib.tools.base import ToolBox, retry_with_backoff


class _Flaky:
    """测试专用：前 n 次调用抛 ToolError，之后成功。"""

    def __init__(self, fail_times: int, result: str = "ok") -> None:
        self.calls = 0
        self.fail_times = fail_times
        self.result = result

    @retry_with_backoff(max_retries=3, base_delay=0.5)
    def run(self) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ToolError("模拟网络抖动")
        return self.result


class TestRetrySemantics:
    def test_succeeds_after_transient_failure(self, monkeypatch):
        """第 1 次失败、第 2 次成功 → 返回结果，总共调用 2 次。"""
        monkeypatch.setattr("time.sleep", lambda _: None)  # 不让测试真睡
        f = _Flaky(fail_times=1)
        assert f.run() == "ok"
        assert f.calls == 2

    def test_exhausted_raises_retry_exhausted(self, monkeypatch):
        """永远失败 → RetryExhaustedError，且它继承 ToolError（调用方语义不变）。"""
        monkeypatch.setattr("time.sleep", lambda _: None)
        f = _Flaky(fail_times=99)
        with pytest.raises(RetryExhaustedError):
            f.run()
        assert f.calls == 4  # 1 次原始调用 + max_retries=3 次重试
        assert issubclass(RetryExhaustedError, ToolError)

    def test_non_tool_error_not_retried(self, monkeypatch):
        """ValueError 不属于重试范围 → 不重试、不包装、原样穿透。"""
        monkeypatch.setattr("time.sleep", lambda _: None)

        @retry_with_backoff(max_retries=3, base_delay=0.5)
        def boom() -> None:
            boom.calls += 1  # type: ignore[attr-defined]
            raise ValueError("程序 bug，重试无意义")

        boom.calls = 0  # type: ignore[attr-defined]
        with pytest.raises(ValueError):
            boom()
        assert boom.calls == 1  # type: ignore[attr-defined]

    def test_max_retries_zero_no_retry(self, monkeypatch):
        """max_retries=0 → 只尝试 1 次，立刻抛。"""
        monkeypatch.setattr("time.sleep", lambda _: None)

        @retry_with_backoff(max_retries=0, base_delay=0.5)
        def boom() -> None:
            boom.calls += 1  # type: ignore[attr-defined]
            raise ToolError("x")

        boom.calls = 0  # type: ignore[attr-defined]
        with pytest.raises(RetryExhaustedError):
            boom()
        assert boom.calls == 1  # type: ignore[attr-defined]

    def test_backoff_delays_double(self, monkeypatch):
        """退避序列翻倍：0.5 → 1.0 → 2.0（sleep 参数可断言 = 设计可测）。"""
        slept: list[float] = []
        monkeypatch.setattr("time.sleep", slept.append)

        @retry_with_backoff(max_retries=3, base_delay=0.5)
        def boom() -> None:
            raise ToolError("x")

        with pytest.raises(RetryExhaustedError):
            boom()
        assert slept == [0.5, 1.0, 2.0]


class TestToolBox:
    def test_mock_mode_returns_mock_data(self):
        """mock 模式：三个工具全部离线返回假数据（M7 离线调试的基础）。"""
        box = ToolBox(Settings(mock_mode=True))
        assert box.search("大模型幻觉").results[0].title.startswith("大模型幻觉")
        assert box.fetch_papers("rag").__class__.__name__ == "list"
        assert box.fetch_papers("rag")[0].paper_id == "2312.10997"
        assert "幻觉" in box.extract("https://example.com/x").text

    def test_get_unknown_tool_raises(self):
        """未知工具名 → ToolError（ToolBox 是权限控制点，名字必须白名单内）。"""
        box = ToolBox(Settings(mock_mode=True))
        with pytest.raises(ToolError):
            box.get("不存在的工具")
