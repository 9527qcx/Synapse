"""嵌入模型：把文本变成向量，语义相似 = 向量接近。

- MockEmbedder：确定性词袋嵌入（不联网、零依赖），测试与离线调试用
- BgeM3Embedder：真实本地嵌入模型（lazy import，只在需要时加载 torch）
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """嵌入器协议：所有嵌入实现都要满足这个形状。"""

    name: str
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_one(self, text: str) -> list[float]: ...


class MockEmbedder:
    """确定性词袋嵌入 —— 256 个桶的比喻。

    工作原理：
    1. 把文本切成 token（用 \\w+ 匹配：中文串/英文单词都是一个 token）
    2. 每个 token 用 blake2b 哈希算出固定桶号（0-255）——同一个词永远进同一个桶
    3. 向量 = 256 个格子的计数，每个词往自己桶里 +1
    4. L2 归一化：向量长度变为 1（余弦相似度 = 点积的前提）

    为什么用 blake2b 而不用内置 hash()：
      内置 hash() 每次进程启动会随机化（防哈希碰撞攻击），
      同一个词两次运行可能进不同桶 → 向量不稳定，测试无法预期。
      blake2b 是确定性的：同一个输入永远同一个输出。

    为什么「词面重叠高的文本余弦大」：
      共享的 token 会落进相同的桶 → 桶分布接近 → 夹角小。
      M6 的去重测试就靠这个特性构造可预期的结果。
    """

    name = "mock"
    dimension = 256
    _NBUCKETS = 256

    def _tokens(self, text: str) -> list[str]:
        # 小写归一（英文大小写视为同词）+ 只保留字母/数字/中文（\w 原生支持中文）
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _bucket(token: str) -> int:
        # blake2b digest_size=1 → 恰好 1 个字节 → 值域 0-255，天然就是桶号
        return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=1).digest())

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._NBUCKETS
        for token in self._tokens(text):
            vector[self._bucket(token)] += 1.0

        # L2 归一化：每维除以向量长度
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            return vector  # 空文本 → 全零向量（长度 0，归一化无意义，原样返回）
        return [x / norm for x in vector]


class BgeM3Embedder:
    """真实本地嵌入模型 BGE-M3（中英文均衡，支持 8K 长文本）。

    注意两点：
    - lazy import：SentenceTransformer/torch 在 __init__ 里才加载，
      保证 mock 模式下永不触碰这两个重库
    - max_seq_length 必须显式设 8192：bge-m3 的卖点就是长文本，
      但默认值是 512，不设就是白买
    """

    name = "bge-m3"
    dimension = 1024

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name, device=device)
        self._model.max_seq_length = 8192

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def get_embedder(settings) -> Embedder:
    """按配置返回嵌入器；local 加载失败时降级 mock（守卫式，业务不中断）。"""
    from ..config.settings import Settings

    settings: Settings
    if settings.embedder == "mock":
        return MockEmbedder()
    try:
        return BgeM3Embedder()
    except Exception as e:  # noqa: BLE001 守卫：模型缺失/下载失败都降级
        logger.warning("BGE-M3 加载失败，降级为 MockEmbedder: %s", e)
        return MockEmbedder()
