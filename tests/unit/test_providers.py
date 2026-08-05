"""models/providers.py 单元测试。

分层测试思路：空 key 保护在工厂层（get_provider），所以 DeepSeekProvider
的「空 key」场景在 test_factory.py 里测，这里只测两个不碰网络的路径：
- MockProvider：确定性 + 关键词路由（纯内存，最该测透）
- OllamaProvider：连接失败 → ProviderError（错误映射路径）
"""
from __future__ import annotations

import json

import pytest

from synapselib.core.errors import ProviderError
from synapselib.models.providers import MockProvider, OllamaProvider
from synapselib.models.schemas import ChatMessage, ChatRole


class TestMockProvider:
    """Mock 的三个关键行为：确定性、关键词路由、只看最后一条 user 消息。"""

    @staticmethod
    def _make(content: str) -> list[ChatMessage]:
        return [ChatMessage(role=ChatRole.USER, content=content)]

    def test_determinism(self):
        """同输入两次调用 → content 完全一致（测试可预期的根基）。"""
        m = MockProvider()
        msg = self._make("请把主题拆解为子查询（输出 json）")
        assert m.complete(msg).content == m.complete(msg).content

    def test_query_keyword_routing(self):
        """含「拆解」→ 返回可解析 JSON，含 queries 字段。"""
        m = MockProvider()
        resp = m.complete(self._make("请把主题拆解为子查询（输出 json）"))
        parsed = json.loads(resp.content)
        assert "queries" in parsed
        assert len(parsed["queries"]) == 4

    def test_extract_keyword_routing(self):
        """含「提取」→ 返回 claims + evidence_excerpt。"""
        m = MockProvider()
        resp = m.complete(self._make("请提取这段文字的 claims（json）"))
        parsed = json.loads(resp.content)
        assert "claims" in parsed
        assert "evidence_excerpt" in parsed
        assert len(parsed["claims"]) == 3

    def test_fallback_when_no_keyword(self):
        """不含任何关键词 → 固定兜底文本（不是 JSON 也能返回）。"""
        m = MockProvider()
        resp = m.complete(self._make("随便说点什么"))
        assert resp.provider == "mock"
        assert "固定" in resp.content

    def test_only_last_user_message_matters(self):
        """路由只看最后一条 user 消息：前面有别的消息不影响。"""
        m = MockProvider()
        messages = [
            ChatMessage(role=ChatRole.USER, content="完全不相关的上一轮内容"),
            ChatMessage(role=ChatRole.ASSISTANT, content="好的"),
            ChatMessage(role=ChatRole.USER, content="请拆解子查询（json）"),
        ]
        resp = m.complete(messages)
        assert "queries" in json.loads(resp.content)


class TestOllamaProvider:
    """Ollama 的连接失败路径：不可达端口 → ProviderError（降级链的信号）。"""

    def test_unreachable_host_raises_provider_error(self):
        p = OllamaProvider(base_url="http://127.0.0.1:9", model="test-model", timeout=1.0)
        with pytest.raises(ProviderError):
            p.complete([ChatMessage(role=ChatRole.USER, content="hello")])
