"""models/embeddings.py 单元测试：MockEmbedder 的确定性、归一化、相似度信号。"""
from __future__ import annotations

import math

import pytest

from synapselib.config.settings import Settings
from synapselib.models.embeddings import MockEmbedder, get_embedder


def _cos(a: list[float], b: list[float]) -> float:
    """余弦相似度：归一化后向量之间就是点积。"""
    return sum(x * y for x, y in zip(a, b))


class TestMockEmbedder:
    def test_dimension(self):
        assert MockEmbedder().dimension == 256

    def test_deterministic(self):
        """同文本两次嵌入结果完全一致（blake2b 跨进程稳定）。"""
        m = MockEmbedder()
        assert m.embed_one("RAG 优于微调") == m.embed_one("RAG 优于微调")

    def test_normalized(self):
        """L2 归一化：向量长度 ≈ 1.0（余弦=点积的前提）。"""
        v = MockEmbedder().embed_one("RAG 优于微调")
        norm = math.sqrt(sum(x * x for x in v))
        assert norm == pytest.approx(1.0)

    def test_similarity_signal(self):
        """词面重叠高的文本余弦 > 无关文本（M6 去重测试的根基）。"""
        m = MockEmbedder()
        similar = _cos(m.embed_one("RAG 优于微调"), m.embed_one("RAG 比微调好"))
        irrelevant = _cos(m.embed_one("RAG 优于微调"), m.embed_one("今天天气不错"))
        assert similar > irrelevant
        assert similar > 0.3   # 共享「rag」「微调」两个 token，余弦应有明显信号
        assert irrelevant < 0.1

    def test_empty_text_returns_zero_vector(self):
        """无 token 的文本 → 全零向量（归一化对零向量无意义，原样返回）。"""
        v = MockEmbedder().embed_one("！！！")
        assert all(x == 0.0 for x in v)

    def test_batch_embed_matches_individual(self):
        """embed() 批量结果 = 逐条 embed_one 的结果。"""
        m = MockEmbedder()
        texts = ["RAG 优于微调", "今天天气不错"]
        assert m.embed(texts) == [m.embed_one(t) for t in texts]


class TestGetEmbedder:
    def test_mock_mode_returns_mock(self):
        assert isinstance(get_embedder(Settings(embedder="mock")), MockEmbedder)

    def test_local_mode_falls_back_to_mock(self, monkeypatch):
        """local 模式加载失败 → 守卫式降级 mock，不抛异常。

        ⚠️ 不能真实触发 BgeM3Embedder：它会下载 2.2GB 模型权重（还可能要联网）。
        测试里用 monkeypatch 把 BgeM3Embedder 换成「构造必炸」的假类，
        模拟「加载失败」环境 —— 这正是上一课讲的：测试不该碰真实外部依赖。
        """
        import synapselib.models.embeddings as emb_module

        class _FailingBgeM3:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("模拟模型加载失败")

        monkeypatch.setattr(emb_module, "BgeM3Embedder", _FailingBgeM3)
        e = get_embedder(Settings(embedder="local"))
        assert isinstance(e, MockEmbedder)
