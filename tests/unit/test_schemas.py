"""core/schemas.py 单元测试：合法构造、边界值、归一化。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from synapselib.core.schemas import (
    CritiqueOutput,
    PublicationStatus,
    ResearchSnippet,
    SourceType,
    SubQueryPlan,
    Task,
    TaskStatus,
    Verdict,
)
from synapselib.core.state import empty_state


class TestTask:
    def test_defaults(self):
        t = Task(description="检索 RLHF 相关论文")
        assert t.status == TaskStatus.PENDING
        assert t.priority == 5
        assert t.source.value == "initial"
        assert t.task_id  # 自动生成

    def test_priority_bounds(self):
        with pytest.raises(ValidationError):
            Task(description="x", priority=0)
        with pytest.raises(ValidationError):
            Task(description="x", priority=11)

    def test_blank_description_rejected(self):
        with pytest.raises(ValidationError):
            Task(description="   ")


class TestResearchSnippet:
    @staticmethod
    def _make(**overrides) -> ResearchSnippet:
        base = dict(
            source_url="https://arxiv.org/abs/2305.10973",
            source_title="A Survey of Hallucination Mitigation",
            source_type=SourceType.PAPER,
            claims=["RLHF 可降低幻觉率", "RAG 引入外部知识"],
            evidence_excerpt="We survey recent approaches...",
            credibility_score=7.5,
            task_id="t1",
        )
        base.update(overrides)
        return ResearchSnippet(**base)

    def test_valid(self):
        s = self._make()
        assert s.publication_status == PublicationStatus.UNKNOWN
        assert s.topic_tags == []
        assert s.snippet_id  # 自动生成

    def test_credibility_upper_bound_rejected(self):
        with pytest.raises(ValidationError):
            self._make(credibility_score=10.1)

    def test_credibility_lower_bound_rejected(self):
        with pytest.raises(ValidationError):
            self._make(credibility_score=-0.5)

    def test_claims_strip_and_drop_blank(self):
        s = self._make(claims=["  RLHF 可降低幻觉率  ", "   ", "RAG 引入外部知识"])
        assert s.claims == ["RLHF 可降低幻觉率", "RAG 引入外部知识"]

    def test_claims_normalized_lowercase(self):
        s = self._make(claims=["RAG 优于 Fine-tuning", "rag 优于 fine-tuning"])
        assert "rag 优于 fine-tuning" in s.claims_normalized

    def test_published_at_invalid_format_rejected(self):
        with pytest.raises(ValidationError):
            self._make(published_at="2025 年 3 月")

    def test_published_at_iso_accepted(self):
        s = self._make(published_at="2025-03-01")
        assert s.published_at == "2025-03-01"

    def test_snippet_id_unique(self):
        assert self._make().snippet_id != self._make().snippet_id


class TestCritiqueOutput:
    def test_valid_with_defaults(self):
        c = CritiqueOutput(verdict=Verdict.PASS)
        assert c.snippet_evaluations == []
        assert c.revision_tasks == []
        assert c.conflict_details == []

    def test_invalid_verdict_rejected(self):
        with pytest.raises(ValidationError):
            CritiqueOutput(verdict="maybe")  # type: ignore[arg-type]


class TestSubQueryPlan:
    def test_empty_queries_rejected(self):
        with pytest.raises(ValidationError):
            SubQueryPlan(queries=[])


class TestAgentState:
    def test_empty_state_fields(self):
        st = empty_state("大模型幻觉缓解技术")
        assert st["research_topic"] == "大模型幻觉缓解技术"
        for key in ("tasks", "research_snippets", "approved_snippets",
                    "memory_context", "retry_records", "user_preferences"):
            assert key in st
        assert st["error_count"] == 0
