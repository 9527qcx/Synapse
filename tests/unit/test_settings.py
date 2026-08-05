"""config/settings.py 单元测试：默认值、环境变量覆盖、路由解析、路径解析。"""
from __future__ import annotations

from pathlib import Path

from synapselib.config.settings import ROOT_DIR, Settings


def _fresh(monkeypatch, **overrides: str) -> Settings:
    """清空全部 SYNAPSE_* 环境变量后构造 Settings（避开宿主机环境影响）。"""
    for key in list(__import__("os").environ):
        if key.startswith("SYNAPSE_"):
            monkeypatch.delenv(key, raising=False)
    for k, v in overrides.items():
        monkeypatch.setenv(f"SYNAPSE_{k}", v)
    return Settings()


class TestDefaults:
    def test_model_defaults(self):
        s = Settings()
        assert s.deepseek_base_url == "https://api.deepseek.com"
        assert s.deepseek_model == "deepseek-chat"
        assert s.mock_mode is False
        assert s.embedder == "local"
        assert s.dedup_cosine_threshold == 0.90
        assert s.dedup_claim_overlap == 0.80

    def test_routes_default_chain(self):
        s = Settings()
        assert s.routes["plan"] == ["deepseek", "ollama", "mock"]
        assert s.routes["extract"] == ["deepseek", "ollama", "mock"]
        assert s.routes["summarizer"] == ["deepseek", "mock"]


class TestEnvOverride:
    def test_mock_mode_flag(self, monkeypatch):
        assert _fresh(monkeypatch, MOCK_MODE="1").mock_mode is True

    def test_custom_route_chain(self, monkeypatch):
        s = _fresh(monkeypatch, ROUTE_PLAN="ollama,deepseek")
        assert s.routes["plan"] == ["ollama", "deepseek"]

    def test_route_chain_ignores_spaces(self, monkeypatch):
        s = _fresh(monkeypatch, ROUTE_PLAN="ollama, deepseek , mock")
        assert s.routes["plan"] == ["ollama", "deepseek", "mock"]


class TestLangfuseEnabled:
    def test_no_keys_disabled(self, monkeypatch):
        s = _fresh(monkeypatch)
        assert s.langfuse_enabled is False

    def test_partial_keys_disabled(self, monkeypatch):
        s = _fresh(monkeypatch, LANGFUSE_PUBLIC_KEY="pk-x")
        assert s.langfuse_enabled is False

    def test_both_keys_enabled(self, monkeypatch):
        s = _fresh(monkeypatch, LANGFUSE_PUBLIC_KEY="pk-x", LANGFUSE_SECRET_KEY="sk-y")
        assert s.langfuse_enabled is True


class TestChromaPath:
    def test_relative_resolves_under_project_root(self):
        s = Settings()
        assert s.chroma_path() == ROOT_DIR / "data" / "chroma"

    def test_absolute_stays_absolute(self, monkeypatch):
        s = _fresh(monkeypatch, CHROMA_DIR="D:/custom/chroma")
        assert s.chroma_path() == Path("D:/custom/chroma")
