"""tools/searxng.py 单元测试：mock 确定性 + 真实模式失败重试链路。

与 M4 的 OllamaProvider 测试同思路：网络路径用不可达端口触发，
重试的 sleep 用 monkeypatch 掐掉，测试既快又不碰外网。
"""
from __future__ import annotations

import pytest

from synapselib.core.errors import RetryExhaustedError
from synapselib.tools.searxng import SearXNGClient


class TestMockMode:
    def test_deterministic(self):
        """同 query 两次调用 → 完全相同（结果有序、内容固定）。"""
        c = SearXNGClient(base_url="http://localhost:8080", mock_mode=True)
        r1, r2 = c.search("大模型幻觉"), c.search("大模型幻觉")
        assert r1 == r2
        assert r1.results[0].url == r2.results[0].url

    def test_query_echoes(self):
        c = SearXNGClient(base_url="http://localhost:8080", mock_mode=True)
        assert c.search("RAG 与微调").query == "RAG 与微调"

    def test_max_results_truncates(self):
        """max_results 限制返回条数（mock 数据共 3 条）。"""
        c = SearXNGClient(base_url="http://localhost:8080", mock_mode=True)
        assert len(c.search("q", max_results=2).results) == 2


class TestRealMode:
    def test_unreachable_host_retries_then_exhausted(self, monkeypatch):
        """不可达端口 → 重试 2 次后抛 RetryExhaustedError（不碰真网络）。"""
        monkeypatch.setattr("time.sleep", lambda _: None)  # 掐掉退避等待
        c = SearXNGClient(base_url="http://127.0.0.1:9", timeout=1.0, mock_mode=False)
        with pytest.raises(RetryExhaustedError):
            c.search("q")
