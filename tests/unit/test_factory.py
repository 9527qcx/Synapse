"""models/factory.py 单元测试：短路、降级、全失败、配置错误四个场景。"""
from __future__ import annotations

import pytest

from synapselib.config.settings import Settings
from synapselib.core.errors import ConfigError, ModelError, ProviderError
from synapselib.models.factory import ModelFactory
from synapselib.models.providers import MockProvider
from synapselib.models.schemas import ChatMessage, ChatRole


class _BrokenProvider:
    """测试专用：必然抛 ProviderError 的假 provider。"""

    name = "broken"

    def complete(self, messages, **kwargs):
        raise ProviderError("故意失败")


def test_fallback_to_mock(monkeypatch):
    """降级链：broken 挂了 → 自动滑到 mock。"""
    def fake_get_provider(self, name):
        if name == "broken":
            return _BrokenProvider()
        return MockProvider()
    monkeypatch.setattr(ModelFactory, "get_provider", fake_get_provider)

    s = Settings(mock_mode=False, route_plan="broken,mock")
    resp = ModelFactory(s).complete("plan", [ChatMessage(role=ChatRole.USER, content="拆解（json）")])
    assert resp.provider == "mock"


def test_mock_mode_short_circuit():
    """mock_mode=True 时不管路由表，直接走 mock（离线调试短路）。"""
    s = Settings(mock_mode=True, route_plan="deepseek")  # 路由表故意指向 deepseek
    resp = ModelFactory(s).complete("plan", [ChatMessage(role=ChatRole.USER, content="拆解（json）")])
    assert resp.provider == "mock"


def test_all_providers_fail_raises_model_error(monkeypatch):
    """降级链全部失败 → ModelError，错误消息包含每一环的失败原因。"""
    def fake_get_provider(self, name):
        return _BrokenProvider()  # 无脑返回必挂的
    monkeypatch.setattr(ModelFactory, "get_provider", fake_get_provider)

    s = Settings(mock_mode=False, route_plan="broken,also_broken")
    with pytest.raises(ModelError) as exc:
        ModelFactory(s).complete("plan", [ChatMessage(role=ChatRole.USER, content="x")])
    assert "broken" in str(exc.value)
    assert "also_broken" in str(exc.value)


def test_unknown_task_kind_raises_config_error():
    """mock_mode=False 时未知 task_kind → ConfigError（配置错误，不是运行时错误）。"""
    s = Settings(mock_mode=False)
    with pytest.raises(ConfigError):
        ModelFactory(s).complete("不存在的任务", [])
