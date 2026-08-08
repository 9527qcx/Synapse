"""memory/vector_store.py 单元测试：写入门槛、去重（粗筛+细判）、检索、主题筛选。

验收标准对照 docs/M6_memory_design.md §5。全部用 MockEmbedder + 临时目录，完全离线。

测试数据设计（MockEmbedder 的 token 粒度是 \\w+ 整串，中文整串算 1 个 token）：
- 去重触发：文本完全相同 → 余弦 1.0，claims 重合 1.0
- 粗筛通过但细判否决：A/B 文本共享 15/16 桶（余弦 0.9375 > 0.90 过粗筛），
  但 claims 重合 1/2 = 0.5 ≤ 0.80 → 细判否决 → 两条并存（宁可多写，不误杀）
"""
from __future__ import annotations

import pytest

from synapselib.config.settings import Settings
from synapselib.core.schemas import ResearchSnippet, SourceType
from synapselib.memory.vector_store import VectorStore
from synapselib.models.embeddings import MockEmbedder

TITLE = "survey"
EVIDENCE = "this survey summarizes the main causes of hallucination in llm agents"


def _snippet(**over):
    """构造测试片段：必填字段给默认值，over 覆盖。"""
    base = dict(
        source_url="https://example.com/a",
        source_title=TITLE,
        source_type=SourceType.PAPER,
        evidence_excerpt=EVIDENCE,
        credibility_score=8.0,
        task_id="t1",
    )
    base.update(over)
    return ResearchSnippet(**base)


@pytest.fixture
def store(tmp_path):
    """每个测试独立的临时 Chroma 目录（测试隔离，互不污染）。"""
    settings = Settings(
        embedder="mock",
        chroma_dir=str(tmp_path / "chroma"),
    )
    return VectorStore(settings, MockEmbedder())


class TestWriteThreshold:
    def test_low_credibility_rejected(self, store):
        """credibility < 7 → rejected，不入库。"""
        r = store.add_snippet(_snippet(credibility_score=5.0))
        assert r.status == "rejected"
        assert "credibility" in r.reason
        assert store.count() == 0

    def test_threshold_boundary_accepted(self, store):
        """credibility = 7（≥ 阈值）→ 写入。"""
        r = store.add_snippet(_snippet(credibility_score=7.0))
        assert r.status == "written"
        assert store.count() == 1


class TestDeduplication:
    def test_identical_text_is_duplicate(self, store):
        """同文本同 claims → 余弦 1.0 过粗筛、重合 1.0 过细判 → duplicate。"""
        first = _snippet(snippet_id="s-aaa", claims=["RAG", "prompt"])
        assert store.add_snippet(first).status == "written"

        dup = _snippet(snippet_id="s-dup", claims=["RAG", "prompt"],
                       source_url="https://example.com/dup")
        r = store.add_snippet(dup)
        assert r.status == "duplicate"
        assert r.duplicate_of == "s-aaa"
        assert r.similarity == pytest.approx(1.0)
        assert store.count() == 1  # 没写进去

    def test_high_cosine_but_low_claim_overlap_written(self, store):
        """余弦 0.9375 > 0.90 过粗筛，但 claims 重合 1/2 ≤ 0.80 → 不判重复。

        教学点：向量相似只是「快而糙」的信号，claims 重合才是「准」的判决。
        """
        a = _snippet(snippet_id="s-a", claims=["RAG", "prompt"])
        assert store.add_snippet(a).status == "written"

        b = _snippet(snippet_id="s-b", claims=["RAG", "debug"],
                     source_url="https://example.com/b")
        r = store.add_snippet(b)
        assert r.status == "written"  # 观点不同 → 宁可多写
        assert store.count() == 2

    def test_empty_claims_never_duplicate(self, store):
        """空 claims → 跳过细判，视为不重复直接写（防丢信息）。"""
        first = _snippet(snippet_id="s-aaa", claims=["RAG", "prompt"])
        store.add_snippet(first)

        empty = _snippet(snippet_id="s-empty", claims=[])
        assert store.add_snippet(empty).status == "written"
        assert store.count() == 2


class TestSearch:
    def test_similarity_signal_and_sorting(self, store):
        """相关片段相似度 > 无关片段，且排序最前。"""
        a = _snippet(snippet_id="s-a", claims=["RAG", "prompt"], topic_tags=["RAG"])
        d = _snippet(snippet_id="s-d", source_title="weather",
                     evidence_excerpt="today the weather is sunny and warm",
                     topic_tags=["天气"])
        store.add_snippet(a)
        store.add_snippet(d)

        hits = store.search("survey hallucination")
        assert len(hits) == 2
        assert hits[0].snippet.snippet_id == "s-a"  # 相关排最前
        assert hits[0].similarity > hits[1].similarity
        assert hits[1].snippet.snippet_id == "s-d"

    def test_topic_filter(self, store):
        """主题筛选下推数据库：只返回该主题的片段。"""
        store.add_snippet(_snippet(snippet_id="s-a", topic_tags=["RAG"]))
        store.add_snippet(_snippet(snippet_id="s-d", source_title="weather",
                                   evidence_excerpt="today the weather is sunny",
                                   topic_tags=["天气"]))

        hits = store.search("weather", topic="天气")
        assert [h.snippet.snippet_id for h in hits] == ["s-d"]

    def test_metadata_list_roundtrip(self, store):
        """claims/topic_tags 元数据往返：list 进 list 出，零转换。"""
        store.add_snippet(_snippet(
            snippet_id="s-a",
            claims=["RAG", "prompt"],
            topic_tags=["RAG"],
            published_at=None,
        ))
        sn = store.search("survey")[0].snippet
        assert sn.claims == ["RAG", "prompt"]   # list 原样回读
        assert sn.topic_tags == ["RAG"]
        assert sn.published_at is None          # 唯一还原转换 "" → None
        assert sn.source_type == SourceType.PAPER  # Enum 往返
        assert sn.task_id == "t1"


class TestBatchAndLifecycle:
    def test_add_batch_mixed_results(self, store):
        """批量返回逐条结果：一条写入、一条被门槛拒绝。"""
        results = store.add_batch([
            _snippet(snippet_id="s-ok", credibility_score=8.0),
            _snippet(snippet_id="s-bad", credibility_score=3.0),
        ])
        assert [r.status for r in results] == ["written", "rejected"]
        assert store.count() == 1

    def test_clear_empties_store(self, store):
        store.add_snippet(_snippet(snippet_id="s-1"))
        assert store.count() == 1
        store.clear()
        assert store.count() == 0
