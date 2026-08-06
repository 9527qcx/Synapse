"""tools/arxiv_tool.py 单元测试：mock 确定性 + SDK 缺失 + 网络失败重试。

覆盖三个路径：
- mock 模式：确定性返回假论文（离线）
- SDK 未安装 → ConfigError（不重试：重试也装不上 SDK）
- 真实模式网络失败 → 重试耗尽 RetryExhaustedError（monkeypatch 内部方法，不碰网络）
"""
from __future__ import annotations

import pytest

from synapselib.core.errors import ConfigError, RetryExhaustedError, ToolError
from synapselib.tools.arxiv_tool import ArxivTool


class TestMockMode:
    def test_returns_fixed_papers(self):
        t = ArxivTool(mock_mode=True)
        papers = t.search("RAG survey")
        assert len(papers) == 2
        assert papers[0].paper_id == "2312.10997"

    def test_deterministic(self):
        t = ArxivTool(mock_mode=True)
        assert t.search("q") == t.search("q")

    def test_max_results_truncates(self):
        t = ArxivTool(mock_mode=True)
        assert len(t.search("q", max_results=1)) == 1

    def test_paper_id_no_version_suffix(self):
        """paper_id 剥掉 arxiv SDK 的版本后缀（"2312.10997v1" → "2312.10997"）。"""
        t = ArxivTool(mock_mode=True)
        assert all("v" not in p.paper_id for p in t.search("q"))


class TestRealMode:
    def test_sdk_missing_raises_config_error(self, monkeypatch):
        """sys.modules 里塞 None = 假装 SDK 没装 → ConfigError 直接抛（不重试）。"""
        monkeypatch.setitem(__import__("sys").modules, "arxiv", None)
        with pytest.raises(ConfigError):
            ArxivTool(mock_mode=False).search("rag")

    def test_network_failure_retries_then_exhausted(self, monkeypatch):
        """内部检索方法抛 ToolError（模拟网络失败）→ 重试 2 次后耗尽。"""
        monkeypatch.setattr("time.sleep", lambda _: None)  # 掐掉退避等待

        def boom(self, query, max_results):
            raise ToolError("模拟网络失败")

        monkeypatch.setattr(ArxivTool, "_search_real", boom)
        with pytest.raises(RetryExhaustedError):
            ArxivTool(mock_mode=False).search("rag")
