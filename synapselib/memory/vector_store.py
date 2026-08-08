"""长期向量记忆（ChromaDB）：写入、去重、检索 —— 大纲 §6.2。

- 存储单元：ResearchSnippet；claims/topic_tags 直接存 list（实测 1.x 支持，
  省 JSON 序列化；$contains 对 list 是精确成员匹配，中文也生效）
- 写入门槛：credibility_score >= settings.credibility_score_threshold（§6.4）
- 去重（§6.2）：粗筛（余弦 > 0.90）→ 细判（claims 重合率 > 0.80，分母 max）
- 主题筛选：where {"topic_tags": {"$contains": topic}} 下推数据库
- 相似度换算：cosine 空间 + 归一化向量 → similarity = 1 - distance
"""
from pydantic import BaseModel
from synapselib.core.schemas import ResearchSnippet
from synapselib.models.embeddings import Embedder
from synapselib.config.settings import Settings
import chromadb

class AddResult(BaseModel):
    """单条写入结果。"""
    snippet_id: str
    status: str                    # "written" | "duplicate" | "rejected"
    duplicate_of: str | None = None  # duplicate 时：被重复的 snippet_id
    similarity: float | None = None  # duplicate 时：粗筛余弦相似度
    reason: str = ""                 # rejected 时：原因（如 "credibility 低于 7"）

class MemoryHit(BaseModel):
    """检索命中。"""
    snippet: ResearchSnippet   # 完整重建（claims 从 JSON 反序列化回来）
    similarity: float          # 余弦相似度（0~1，越大越相似）

class VectorStore:
    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        """DI：settings 提供路径与阈值，embedder 提供向量（不自己 new）。"""
        self.settings = settings
        self.embedder = embedder
        client = chromadb.PersistentClient(path=str(settings.chroma_path()))
        # 每次启动用同一个 path → 数据持久化；不存在会自动创建目录
        self.collection = client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},  
        )

    def add_snippet(self, snippet: ResearchSnippet) -> AddResult:
        """写入一个片段。"""
        if snippet.credibility_score < self.settings.credibility_score_threshold:
            return AddResult(
                snippet_id=snippet.snippet_id,
                status="rejected",
                reason=f"credibility 低于阈值 {self.settings.credibility_score_threshold}"
            )
        text = f"{snippet.source_title}。{' '.join(snippet.claims)}。{snippet.evidence_excerpt}"
        vec = self.embedder.embed_one(text)
        # 候选召回（⚠️ 参数名是复数 query_embeddings，且要包成列表 —— 报错会告诉你拼错）
        result = self.collection.query(query_embeddings=[vec], n_results=5)
        # 永远取 [0] —— 外层是 batch 维度（一次查多条才用外层索引）
        ids, dists, metas = result["ids"][0], result["distances"][0], result["metadatas"][0]

        candidates = []
        for i, d in enumerate(dists):
            similarity = 1 - d  # 坑④：cosine 空间 + 归一化向量 → 1 - distance
            if similarity > self.settings.dedup_cosine_threshold:  # 0.90
                candidates.append(i)  # 记下标，细判时按 index 取 claims

        # claims_normalized 是 ResearchSnippet 现成的 property（M2 就写好了，直接用！）
        new_claims = snippet.claims_normalized
        if new_claims:  # 空 claims → 跳过细判，视为不重复（防丢信息）
            for i in candidates:
                cand_claims = metas[i]["claims"]  # 直接是 list（实测 ChromaDB 1.x 支持 list metadata）
                cand_set = {c.strip().lower() for c in cand_claims}
                overlap = len(new_claims & cand_set) / max(len(cand_set), len(new_claims))
                if overlap > self.settings.dedup_claim_overlap:  # 0.80
                    return AddResult(
                        snippet_id=snippet.snippet_id,
                        similarity=1-dists[i],
                        duplicate_of=ids[i],
                        status="duplicate",
                    )
        
        # metadata 按 ResearchSnippet 字段顺序填写（schemas.py §17.2 的定义顺序）
        # claims / topic_tags 直接存 list（实测 ChromaDB 1.x 支持 list 值，省 JSON 序列化，
        # 且 $contains 能对 list 做精确成员匹配）
        meta = {
            "snippet_id": snippet.snippet_id,
            "source_url": snippet.source_url,
            "source_title": snippet.source_title,
            "source_type": snippet.source_type.value,      # ⚠️ Enum 必须存 .value
            "publication_status": snippet.publication_status.value,
            "evidence_excerpt": snippet.evidence_excerpt,
            "credibility_score": snippet.credibility_score,
            "published_at": snippet.published_at or "",     # ⚠️ None 不能存，转空串
            "extracted_at": snippet.extracted_at,
            "task_id": snippet.task_id,
        }
        if snippet.topic_tags:
            meta["topic_tags"] = snippet.topic_tags
        if snippet.claims:
            meta["claims"] = snippet.claims
        self.collection.upsert(
            ids=[snippet.snippet_id],
            embeddings=[vec],
            metadatas=[meta],
            documents=[text],
            )
        return AddResult(
            snippet_id=snippet.snippet_id,
            status="written",
        )
            
        
        


    def add_batch(self, snippets: list[ResearchSnippet]) -> list[AddResult]: 
        """批量写入多个片段。"""
        return [self.add_snippet(snippet) for snippet in snippets]
    
    def search(self, query: str, top_k: int = 5, topic: str | None = None) -> list[MemoryHit]: 
        """检索 top_k 个最相关的片段。"""
        vec = self.embedder.embed_one(query)
        kwargs = {}
        if topic:
            # $contains 对 list 值是精确成员匹配（实测：中文也生效）→ 下推到数据库过滤
            kwargs["where"] = {"topic_tags": {"$contains": topic}}
        result = self.collection.query(query_embeddings=[vec], n_results=top_k, **kwargs)
        ids, dists, metas = result["ids"][0], result["distances"][0], result["metadatas"][0]

        hits = []
        for i in range(len(ids)):
            meta = dict(metas[i])
            meta["published_at"] = meta["published_at"] or None  # 唯一的还原转换（"" → None）
            hits.append(MemoryHit(
                snippet=ResearchSnippet(**meta),
                similarity=1-dists[i],
            ))
        return hits

    def count(self) -> int:       # 已写入的片段数（测试/统计用）
        return self.collection.count()
    def clear(self) -> None:      # 清空（测试隔离用，别在业务里调用）
        ids = self.collection.get()["ids"]
        if ids:
            self.collection.delete(ids=ids)