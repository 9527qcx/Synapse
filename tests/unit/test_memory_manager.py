"""memory/manager.py 单元测试：五个委托方法 + mock 模式全链路离线可用。

M7 只跟 MemoryManager 打交道，所以这里测的是「外部门面」：
- remember / recall → 长期记忆（vector_store 委托）
- add_turn / context / summarize_if_needed → 短期记忆（stm 委托）
- preferences → 用户偏好
- settings 配置要真正流进子模块（budget_tokens 等不写死）
"""
from __future__ import annotations

import pytest

from synapselib.config.settings import Settings
from synapselib.core.schemas import ResearchSnippet, SourceType
from synapselib.memory.manager import MemoryManager
from synapselib.memory.preferences import UserPreferences


def _snippet(**over):
    base = dict(
        source_url="https://example.com/a",
        source_title="survey",
        source_type=SourceType.PAPER,
        evidence_excerpt="this survey summarizes hallucination causes",
        credibility_score=8.0,
        task_id="t1",
    )
    base.update(over)
    return ResearchSnippet(**base)


@pytest.fixture
def manager(tmp_path):
    """MockEmbedder + 临时 Chroma 目录 + 小预算 stm：全离线，预算可控。"""
    settings = Settings(
        embedder="mock",
        chroma_dir=str(tmp_path / "chroma"),
        preferences_file=str(tmp_path / "prefs.json"),
        stm_budget_tokens=100,  # 120 tokens 就会触发压缩，便于测试
        stm_keep_last_turns=2,
    )
    return MemoryManager(settings)


class TestDelegation:
    def test_remember(self, manager):
        r = manager.remember(_snippet())
        assert r.status == "written"

    def test_recall_finds_remembered(self, manager):
        """remember 写的片段能被 recall 召回（含主题筛选）。"""
        manager.remember(_snippet(snippet_id="s-1", claims=["RAG"], topic_tags=["RAG"]))
        hits = manager.recall("survey", topic="RAG")
        assert [h.snippet.snippet_id for h in hits] == ["s-1"]
        assert hits[0].similarity > 0

    def test_context_contains_turns(self, manager):
        manager.add_turn("user", "帮我研究幻觉")
        ctx = manager.context()
        assert "帮我研究幻觉" in ctx
        assert "【最近对话】" in ctx

    def test_summarize_if_needed_compresses(self, manager):
        """超预算且轮数够 → 压缩，摘要进入上下文。"""
        manager.add_turn("user", "x" * 80)      # 40 tokens
        manager.add_turn("assistant", "y" * 80)  # 累计 80
        manager.add_turn("user", "z" * 80)       # 累计 120 > 100，3 轮 > 保留 2 轮
        manager.summarize_if_needed(lambda text: "摘要")
        ctx = manager.context()
        assert "【历史摘要】" in ctx and "摘要" in ctx

    def test_preferences_object(self, manager):
        prefs = manager.preferences()
        assert isinstance(prefs, UserPreferences)
        prefs.update({"depth_preference": "深入"})
        assert manager.preferences().get("depth_preference") == "深入"  # 同一实例

    def test_settings_wired_to_stm(self, manager):
        """settings 的预算配置要真正流进短期记忆（不写死默认值）。"""
        assert manager.stm.budget_tokens == 100
        assert manager.stm.keep_last_turns == 2


class TestOfflineChain:
    def test_full_chain_offline(self, manager):
        """mock 模式：写入 → 检索 → 上下文 → 偏好，全程不碰网络不碰真实模型。"""
        manager.remember(_snippet(snippet_id="s-1", claims=["RAG"], topic_tags=["RAG"]))
        assert manager.recall("survey")[0].snippet.snippet_id == "s-1"

        manager.add_turn("user", "hello")
        assert "hello" in manager.context()

        manager.preferences().update({"report_style": "markdown"})
        assert manager.preferences().get("report_style") == "markdown"