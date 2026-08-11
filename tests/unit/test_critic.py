"""agents/critic.py 单元测试：评分规则表 + 评分防线 + 交叉验证。

对应 M7 设计文档 §6 验收标准：
- score_credibility 规则表：paper 满时效 → 10.0；blog + 未知时间 → 低分；边界值精确
- LLM 说 PASS 但某片段评分 < 7 → 强制 NEEDS_REVISION + revision_tasks 非空
- 全部高分 → PASS 原样通过
- 交叉验证统计：同 claims 不同来源的片段 corroborations ≥ 2
"""
from __future__ import annotations

from datetime import datetime, timedelta

from synapselib.agents.critic import Critic, score_credibility
from synapselib.core.schemas import (
    ResearchSnippet,
    SourceType,
    Task,
    Verdict,
)
from synapselib.memory.manager import MemoryManager


def _days_ago(n: int) -> str:
    """n 天前的日期字符串（避免固定日期随时间漂移）。"""
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")


def _snippet(sid, stype=SourceType.PAPER, claims=None, published_at=None,
             url=None):
    return ResearchSnippet(
        snippet_id=sid,
        source_url=url or f"https://example.com/{sid}",
        source_title=sid,
        source_type=stype,
        claims=claims or ["RAG 引入外部知识减少幻觉"],
        evidence_excerpt="（mock）检索增强可显著降低幻觉发生率。",
        credibility_score=10.0,
        topic_tags=["大模型幻觉缓解"],
        published_at=published_at,
        task_id="t1",
    )


def _critic(mock_settings, llm):
    return Critic(mock_settings, MemoryManager(mock_settings), llm=llm)


class TestScoreCredibility:
    """§7.2 加权规则表（60/30/10）精确值。"""

    def test_paper_fresh_full(self):
        assert score_credibility(SourceType.PAPER, _days_ago(730), 1) == 10.0

    def test_three_sources_clamped_to_max(self):
        """3 源加权后 16 分 → clamp 到 10（0-10 契约）。"""
        assert score_credibility(SourceType.PAPER, _days_ago(730), 3) == 10.0

    def test_two_sources_news_clamped(self):
        """news + 2 源也超 10 → clamp。"""
        assert score_credibility(SourceType.NEWS, None, 2) == 10.0

    def test_blog_unknown_low(self):
        """blog + 未知时间 + 单源：最低档组合。"""
        assert score_credibility(SourceType.BLOG, None, 1) == 5.9

    def test_freshness_2_3_years(self):
        """731 天（2.0+ 年）→ 时效 7 分。"""
        assert score_credibility(SourceType.PAPER, _days_ago(731), 1) == 9.7

    def test_freshness_3_4_years_flat(self):
        """3-4 年衰减第一档仍 7 分。"""
        assert score_credibility(SourceType.PAPER, _days_ago(1100), 1) == 9.7

    def test_freshness_4_5_years_decay(self):
        """4.5 年 → 时效衰减到 5 分。"""
        assert score_credibility(SourceType.PAPER, _days_ago(1643), 1) == 9.5

    def test_none_treated_as_unknown(self):
        """None 日期 → 时效 5 分（不能炸）。"""
        assert score_credibility(SourceType.PAPER, None, 1) == 9.5

    def test_invalid_date_treated_as_unknown(self):
        """非法日期字符串 → 走 except → 时效 5 分。"""
        assert score_credibility(SourceType.PAPER, "not-a-date", 1) == 9.5


class TestDefenseLine:
    """硬规则覆盖软判断：LLM 的裁决可以被规则推翻。"""

    def test_pass_overridden_by_low_score(self, mock_settings, make_llm):
        """LLM 说 PASS 但 blog 只有 5.9 → 强制 NEEDS_REVISION + 修订任务。"""
        c = _critic(mock_settings, make_llm(verdicts=["PASS"]))
        out = c.critique(Task(description="大模型幻觉缓解"),
                         [_snippet("s1", SourceType.BLOG, published_at=None)])
        assert out.verdict == Verdict.NEEDS_REVISION
        assert out.revision_tasks, "强制返工必须带修订任务"

    def test_pass_kept_when_all_high(self, mock_settings, make_llm):
        """全部高分 → PASS 原样通过。"""
        c = _critic(mock_settings, make_llm(verdicts=["PASS"]))
        out = c.critique(Task(description="大模型幻觉缓解"),
                         [_snippet("s1", published_at=_days_ago(100))])
        assert out.verdict == Verdict.PASS
        assert out.revision_tasks == []

    def test_llm_score_overwritten(self, mock_settings, make_llm):
        """LLM 给 9.0 → 防线覆盖成规则分 10.0。"""
        llm = make_llm(verdicts=["PASS"], evals=[{
            "snippet_id": "s1", "credibility_score": 9.0, "issues": [], "suggestion": ""}])
        c = _critic(mock_settings, llm)
        out = c.critique(Task(description="大模型幻觉缓解"),
                         [_snippet("s1", published_at=_days_ago(100))])
        evals = {e.snippet_id: e for e in out.snippet_evaluations}
        assert evals["s1"].credibility_score == 10.0

    def test_missing_evaluation_filled(self, mock_settings, make_llm):
        """LLM 漏评的片段 → 防线补评估（规则分）。"""
        c = _critic(mock_settings, make_llm(verdicts=["PASS"]))  # evals 为空
        out = c.critique(Task(description="大模型幻觉缓解"),
                         [_snippet("s1"), _snippet("s2", SourceType.BLOG)])
        evals = {e.snippet_id: e for e in out.snippet_evaluations}
        assert set(evals) == {"s1", "s2"}, "两个片段都必须有评估"


class TestCorroboration:
    def test_same_claims_two_sources_boosts(self, mock_settings, make_llm):
        """同 claims + 不同 URL → corroborations=2 → cross 20 分。

        blog 单源 5.9，双源 0.6*4 + 0.3*20 + 0.1*5 = 8.9 —— 交叉验证把分抬过门槛。
        """
        c = _critic(mock_settings, make_llm(verdicts=["PASS"]))
        out = c.critique(
            Task(description="大模型幻觉缓解"),
            [_snippet("s1", SourceType.BLOG, claims=["同一主张"], published_at=None),
             _snippet("s2", SourceType.BLOG, claims=["同一主张"], published_at=None,
                      url="https://example.com/other")])
        evals = {e.snippet_id: e for e in out.snippet_evaluations}
        assert evals["s1"].credibility_score == 8.9

    def test_same_source_not_counted(self, mock_settings, make_llm):
        """同 URL 不算第二来源：两条同 URL 片段 corroborations 仍为 1。"""
        c = _critic(mock_settings, make_llm(verdicts=["PASS"]))
        out = c.critique(
            Task(description="大模型幻觉缓解"),
            [_snippet("s1", SourceType.BLOG, claims=["同一主张"], published_at=None,
                      url="https://example.com/same"),
             _snippet("s2", SourceType.BLOG, claims=["同一主张"], published_at=None,
                      url="https://example.com/same")])
        evals = {e.snippet_id: e for e in out.snippet_evaluations}
        assert evals["s1"].credibility_score == 5.9, "同源重合不抬分"
