"""agents/researcher.py 单元测试：复用短路 / 搜索终止 / 容错链 / 去重与门槛。

对应 M7 设计文档 §6 验收标准：
- 正常流程：搜索 → 提取 → 提炼 → 写入，计数正确
- 复用优先：高相似记忆命中 → 记入 reused，不再调用工具（短路）
- 连续 3 次搜索无结果 → errors 上报 + 提前终止
- 提取失败 / 提炼解析失败 → 记 errors，继续其他结果
- duplicate 计入 duplicates_skipped；低可信度 → rejected
"""
from __future__ import annotations

from synapselib.agents.researcher import Researcher
from synapselib.config.settings import Settings
from synapselib.core.schemas import ResearchSnippet, SourceType, Task
from synapselib.memory.manager import MemoryManager
from synapselib.tools.base import ToolBox


def _researcher(settings, tools, llm):
    return Researcher(settings, tools, MemoryManager(settings), llm=llm)


def _task(desc="大模型幻觉缓解", queries=None):
    return Task(description=desc, search_queries=queries or [desc])


class TestHappyPath:
    def test_full_pipeline_counts(self, mock_settings, make_llm):
        """真实 mock 工具全链路：arxiv 1 条 written，blog 2 条 rejected。"""
        r = _researcher(mock_settings, ToolBox(mock_settings), make_llm())
        result = r.research(_task())
        assert result.snippets_written, "arxiv 结果应写入记忆"
        assert len(result.snippets_written) == 1
        assert result.snippets_rejected == 2, "blog 初评 4 分应被门槛拒收"
        assert result.duplicates_skipped == 0
        assert result.errors == []
        s = result.snippets_written[0]
        assert s.source_type == SourceType.PAPER
        assert s.credibility_score == 10.0
        assert s.topic_tags == ["大模型幻觉缓解"]
        assert s.published_at == "2025-01-15"


class TestReuse:
    def test_reuse_shortcuts_search(self, mock_settings, make_llm, make_tools):
        """记忆里已有高相似片段 → 短路：工具一次都不调，记入 reused。"""
        llm, tools = make_llm(), make_tools()
        memory = MemoryManager(mock_settings)
        r = Researcher(mock_settings, tools, memory, llm=llm)
        desc = "大模型幻觉缓解"
        # 直接写入一条嵌入文本与 desc 完全重合的片段（余弦 1.0 ≥ 0.75）
        memory.remember(ResearchSnippet(
            source_url="https://example.com/old", source_title=desc,
            source_type=SourceType.PAPER, claims=[desc], evidence_excerpt=desc,
            credibility_score=8.0, topic_tags=[desc], task_id="t0"))
        result = r.research(_task(desc))
        assert tools.search_calls == 0, "复用命中后不应再搜索"
        assert tools.extract_calls == 0
        assert len(result.snippets_reused) == 1
        assert result.snippets_reused[0].source_url == "https://example.com/old"
        assert not result.snippets_written
        assert result.errors == []

    def test_low_similarity_not_reused(self, mock_settings, make_llm, make_tools):
        """记忆里只有低相似片段 → 不短路，照常搜索。"""
        llm, tools = make_llm(), make_tools()
        memory = MemoryManager(mock_settings)
        r = Researcher(mock_settings, tools, memory, llm=llm)
        # 嵌入文本与 desc 完全不同（claim 用英文，desc 是中文）
        memory.remember(ResearchSnippet(
            source_url="https://example.com/old", source_title="unrelated title",
            source_type=SourceType.PAPER, claims=["totally different claim"],
            evidence_excerpt="nothing in common", credibility_score=8.0,
            topic_tags=["other"], task_id="t0"))
        result = r.research(_task())
        assert tools.search_calls == 1, "低相似不命中 → 正常搜索"
        assert result.snippets_reused == []


class TestEarlyTermination:
    def test_three_empty_searches(self, mock_settings, make_llm, make_tools):
        """连续 3 次搜索无结果 → errors 上报 + 提前终止。"""
        r = _researcher(mock_settings, make_tools(empty=True), make_llm())
        task = Task(description="x", search_queries=["q1", "q2", "q3"])
        result = r.research(task)
        assert "连续 3 次搜索无结果" in result.errors
        assert result.snippets_written == []
        assert result.duplicates_skipped == 0


class TestErrorResilience:
    def test_extract_failure_continues(self, mock_settings, make_llm, make_tools):
        """提取失败 → 记 errors，继续处理其他结果。"""
        r = _researcher(mock_settings, make_tools(fail_extract=True), make_llm())
        result = r.research(_task(queries=["q1", "q2", "q3"]))
        assert len(result.errors) == 3, "3 个结果全部提取失败"
        assert all("提取失败" in e for e in result.errors)
        assert result.snippets_written == []

    def test_parse_failure_continues(self, mock_settings, make_llm):
        """提炼解析失败 → 记 errors，不中断。"""
        r = _researcher(mock_settings, ToolBox(mock_settings), make_llm(bad_extract=True))
        result = r.research(_task())
        assert len(result.errors) == 3, "3 个结果全部提炼解析失败"
        assert all("提炼失败" in e for e in result.errors)
        assert result.snippets_written == []


class TestDedupAndThreshold:
    def test_duplicate_skipped(self, mock_settings, make_llm):
        """同一 mock 数据二次研究：arxiv 条被去重，blog 条仍被门槛拒收。"""
        r = _researcher(mock_settings, ToolBox(mock_settings), make_llm())
        task = _task()
        first = r.research(task)
        assert len(first.snippets_written) == 1
        second = r.research(task)
        assert second.duplicates_skipped == 1, "arxiv 条与首次相同 → duplicate"
        assert second.snippets_rejected == 2
        assert second.snippets_written == []
