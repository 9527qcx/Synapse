"""tools/schemas.py 单元测试：工具契约的校验规则。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from synapselib.tools.schemas import ExtractionResult, PaperMeta, SearchResponse, SearchResult


class TestSearchResult:
    def test_title_not_blank(self):
        """title 是搜索结果的锚点，空标题直接拒绝。"""
        with pytest.raises(ValidationError):
            SearchResult(title="   ", url="https://example.com")

    def test_snippet_and_date_default(self):
        r = SearchResult(title="T", url="https://example.com")
        assert r.snippet == ""
        assert r.published_at is None


class TestSearchResponse:
    def test_default_empty_results(self):
        resp = SearchResponse(query="q")
        assert resp.results == []

    def test_query_echoes(self):
        resp = SearchResponse(query="RAG", results=[SearchResult(title="T", url="https://x")])
        assert resp.query == "RAG"
        assert resp.results[0].title == "T"


class TestPaperMeta:
    def test_defaults(self):
        p = PaperMeta(paper_id="2401.00001", title="T", url="https://arxiv.org/abs/2401.00001")
        assert p.abstract == ""
        assert p.authors == []
        assert p.published_at is None


class TestExtractionResult:
    def test_truncated_default_false(self):
        r = ExtractionResult(url="https://example.com", text="正文")
        assert r.truncated is False
