"""tools/web_extractor.py 单元测试：mock + 截断 + 失败链路。

「假 trafilatura」技巧：monkeypatch 把 sys.modules 里的 trafilatura 换成
一个假模块（extract 返回固定文本），就能离线测到真实模式的完整管线
（下载 → 解析 → 截断），不用真装/真调 trafilatura。
"""
from __future__ import annotations

import pytest

from synapselib.core.errors import ConfigError, RetryExhaustedError, ToolError
from synapselib.tools.web_extractor import WebExtractor


class TestMockMode:
    def test_returns_fixed_text(self):
        e = WebExtractor(mock_mode=True)
        r = e.extract("https://example.com/llm-hallucination-survey")
        assert r.url == "https://example.com/llm-hallucination-survey"
        assert "幻觉" in r.text
        assert r.truncated is False

    def test_deterministic(self):
        e = WebExtractor(mock_mode=True)
        assert e.extract("https://x.com/a") == e.extract("https://x.com/a")


class TestRealMode:
    def test_trafilatura_missing_raises_config_error(self, monkeypatch):
        """SDK 未装 → ConfigError 直接抛（不重试：重试也装不上 SDK）。"""
        monkeypatch.setitem(__import__("sys").modules, "trafilatura", None)
        with pytest.raises(ConfigError):
            WebExtractor(mock_mode=False).extract("https://example.com")

    def test_download_failure_retries_then_exhausted(self, monkeypatch):
        """下载失败（ToolError）→ 重试 2 次后 RetryExhaustedError。"""
        monkeypatch.setattr("time.sleep", lambda _: None)

        def boom(self, url):
            raise ToolError("模拟下载失败")

        monkeypatch.setattr(WebExtractor, "_download", boom)
        with pytest.raises(RetryExhaustedError):
            WebExtractor(mock_mode=False).extract("https://example.com")

    def test_full_pipeline_with_fake_trafilatura(self, monkeypatch):
        """离线走通真实管线：假下载 + 假 trafilatura → 正确的 ExtractionResult。"""
        monkeypatch.setattr("time.sleep", lambda _: None)
        monkeypatch.setattr(
            WebExtractor, "_download", lambda self, url: ("<html><body><p>正文内容</p></body></html>", "测试标题")
        )

        fake = type("FakeTrafilatura", (), {"extract": staticmethod(lambda html, **kw: "正文内容")})
        monkeypatch.setitem(__import__("sys").modules, "trafilatura", fake)

        r = WebExtractor(mock_mode=False).extract("https://example.com/a")
        assert r.title == "测试标题"
        assert r.text == "正文内容"
        assert r.truncated is False

    def test_long_text_truncated(self, monkeypatch):
        """正文超过 max_chars → 截断并标记 truncated=True。"""
        monkeypatch.setattr("time.sleep", lambda _: None)
        long_text = "长" * 1000
        monkeypatch.setattr(WebExtractor, "_download", lambda self, url: ("<html></html>", "T"))
        fake = type("FakeTrafilatura", (), {"extract": staticmethod(lambda html, **kw: long_text)})
        monkeypatch.setitem(__import__("sys").modules, "trafilatura", fake)

        r = WebExtractor(mock_mode=False, max_chars=100).extract("https://example.com/a")
        assert r.truncated is True
        assert len(r.text) == 100
