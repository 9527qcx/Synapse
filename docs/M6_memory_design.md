# M6 记忆系统设计（规划稿）

> 依据大纲 §6（记忆系统设计）、§6.4（写入与淘汰）、§6.5（用户偏好）。
> 阅读顺序建议：先读 §3 算法讲解（理解为什么），再对照 §2 契约写代码。

## 1. 模块结构与职责

```
synapselib/memory/
├── __init__.py        # 包说明
├── vector_store.py    # 长期向量记忆（ChromaDB）：写入、去重、检索、主题筛选  ← 核心
├── short_term.py      # 短期记忆（会话级）：token 预算、滑动窗口、摘要压缩     ← 核心
├── preferences.py     # 用户偏好（JSON 文件）：读取、更新                     ← 简单
└── manager.py         # MemoryManager：总入口，M7 只跟它打交道
```

**依赖（已就绪，直接用）**：
- `core/schemas.py`：`ResearchSnippet`（存储单元）、`ReusedHit`（复用命中）
- `core/errors.py`：`MemoryError`（记忆层异常基类，已定义）
- `models/embeddings.py`：`get_embedder(settings)` → MockEmbedder（mock 模式离线）/ BGE-M3
- `config/settings.py`：`chroma_dir`、`dedup_cosine_threshold=0.90`、`dedup_claim_overlap=0.80`、`stm_budget_tokens=8000`、`stm_keep_last_turns=6`

**mock 模式怎么工作**：MockEmbedder（256 维、确定性）+ 真实 ChromaDB（本地文件）= 完全离线可测的完整记忆链路。MockEmbedder 的余弦相似度信号已在 M4 验证过（相似文本 ~0.5，无关文本 ~0.0）。

---

## 2. 接口契约

### 2.1 vector_store.py —— 长期向量记忆（§6.2）

```python
from pydantic import BaseModel
from synapselib.core.schemas import ResearchSnippet

class AddResult(BaseModel):
    """单条写入结果。"""
    snippet_id: str
    status: str                    # "written" | "duplicate" | "rejected"
    duplicate_of: str | None = None  # duplicate 时：被重复的 snippet_id
    similarity: float | None = None  # duplicate 时：粗筛余弦相似度
    reason: str = ""                 # rejected 时：原因（如 "credibility 低于 7"）

class MemoryHit(BaseModel):
    """检索命中。"""
    snippet: ResearchSnippet   # 完整重建（claims/topic_tags 存的就是 list，原样回读）
    similarity: float          # 余弦相似度（0~1，越大越相似）

class VectorStore:
    def __init__(self, settings, embedder) -> None:
        """DI：settings 提供路径与阈值，embedder 提供向量（不自己 new）。"""

    def add_snippet(self, snippet: ResearchSnippet) -> AddResult: ...
    def add_batch(self, snippets: list[ResearchSnippet]) -> list[AddResult]: ...
    def search(self, query: str, top_k: int = 5, topic: str | None = None) -> list[MemoryHit]: ...
    def count(self) -> int: ...      # 已写入的片段数（测试/统计用）
    def clear(self) -> None: ...     # 清空（测试隔离用，别在业务里调用）
```

**collection 细节**（照着写，别踩坑）：
- 名字：`"snippets"`
- 空间：**必须显式指定 cosine** —— `get_or_create_collection(name, metadata={"hnsw:space": "cosine"})`。ChromaDB 默认是 L2 空间，distance 含义完全不同（坑 ①）
- 永远**显式传 embeddings**（add/upsert/query 都传），不要依赖 collection 的默认嵌入函数——它首次调用会下载 ONNX 模型（坑 ②，离线会挂）
- 余弦换算：归一化嵌入 + cosine 空间 → `distance = 1 - cosine`，所以 `similarity = 1 - distance`

### 2.2 short_term.py —— 短期记忆（§6.1）

```python
class ShortTermMemory:
    def __init__(self, budget_tokens: int = 8000, keep_last_turns: int = 6,
                 summary_tokens: int = 2000) -> None:
        self._turns: list[dict] = []   # [{"role": ..., "content": ...}]
        self._summary: str = ""        # 历史摘要

    def add(self, role: str, content: str) -> None: ...
    def token_count(self) -> int: ...                # 含摘要与全部轮次
    def should_compress(self) -> bool: ...           # 超预算 → True
    def compress(self, summarize: Callable[[str], str]) -> None: ...
    def context(self) -> str: ...                    # "【历史摘要】…【最近对话】…"
    def to_dict(self) -> dict: ...                   # 持久化
    @classmethod
    def from_dict(cls, data: dict) -> "ShortTermMemory": ...
```

**token 估算**：没有真实 tokenizer，用近似 `len(text) // 2`（中文约 1.5 字/token）。滑动窗口是**近似管理**，不追求精确（教学点：宁粗略勿复杂）。

**compress 语义**（§6.1：保留最近 N 轮完整 + 历史摘要）：
1. 旧轮 = 除最近 `keep_last_turns` 轮外的全部轮次
2. 旧轮文本交给 `summarize`（外部注入，M7 时用 ModelFactory 的 summarizer 路由）
3. 新摘要 + 保留轮次的 token 数仍超预算 → 摘要截断到 `summary_tokens`
4. 空轮次不压（没内容没必要调 LLM）

### 2.3 preferences.py —— 用户偏好（§6.5）

```python
class UserPreferences:
    def __init__(self, path: Path) -> None:   # DI：JSON 文件路径
    def get(self, key: str, default=None): ...
    def update(self, feedback: dict) -> None:   # 与现有偏好合并后写盘
    def to_dict(self) -> dict: ...
```

- MVP 只做显式反馈：键是 `preferred_sources`、`depth_preference`、`research_interests`、`report_style`
- 写盘：**临时文件 + os.replace 原子替换**（防写一半断电损坏 JSON，教学点）
- 文件不存在 → 返回空偏好，不要崩

### 2.4 manager.py —— 总入口

```python
class MemoryManager:
    """记忆系统唯一入口。M7 Researcher 只 import 它。"""
    def __init__(self, settings, embedder=None) -> None:
        # embedder 缺省 get_embedder(settings)；组合三个子模块
    def remember(self, snippet: ResearchSnippet) -> AddResult: ...   # 委托 vector_store
    def recall(self, query: str, top_k: int = 5, topic: str | None = None) -> list[MemoryHit]: ...
    def context(self) -> str: ...                                    # 短期记忆上下文
    def summarize_if_needed(self, summarize: Callable[[str], str]) -> None: ...
    def preferences(self) -> UserPreferences: ...
```

---

## 3. 关键算法（先理解为什么，再写）

### 3.1 去重：粗筛 → 细判（§6.2，核心中的核心）

```
add_snippet(snippet):
    # 门槛（§6.4）：可信度 < 7 直接拒绝，连查询都不用做
    if snippet.credibility_score < 7:  → rejected("credibility 低于 7")

    # 1) 嵌入 + 候选召回：拿 snippet 自身向量在库里找最像的 5 个
    vec = embedder.embed_one(snippet 的正文文本)
    hits = collection.query(query_embeddings=[vec], n_results=5)

    # 2) 粗筛：相似度 > 0.90 才进入细判（大部分无关片段在这里被扔掉）
    candidates = [h for h in hits if similarity(h) > dedup_cosine_threshold]

    # 3) 细判：claims 重合率 > 80% 才算真重复
    for c in candidates:
        overlap = |snippet.claims_normalized ∩ c.claims_normalized| / max(len(a), len(b))
        if overlap > dedup_claim_overlap:
            → duplicate(duplicate_of=c.snippet_id, similarity=cosine)

    # 4) 不重复 → 写入（id=snippet_id，metadata 见 2.1，claims/topic_tags 存 list）
    collection.upsert(...)
    → written
```

**为什么先粗筛再细判**：向量相似度是「快而糙」的信号（词面+语义近似），claims 重合是「准而贵」的信号（要逐条比对）。先花几毫秒把 99% 无关的过滤掉，只对最像的几个做精细比对——这是经典的两阶段检索（recall → precision）。

**为什么分母用 max**：要求「短 claims 集合的内容基本被长集合覆盖」才算重复。比如新片段 2 条 claims 全在旧片段 8 条里，重合率 = 2/8 = 25% → 不重复（旧片段信息更多，新片段视角值得保留）。分母用 max 是**保守策略：宁可多写，不误杀**（信息丢失不可逆，冗余可去重）。

**空 claims 怎么办**：`claims` 为空的片段跳过细判、视为不重复直接写（防丢信息）。

### 3.2 短期记忆压缩（§6.1）

```
对话轮次追加 → 每次 add 后自查：
    如果 token_count() > budget_tokens:
        → 需要压缩（should_compress() = True）

compress(summarize):
    旧轮 = turns[:-keep_last_turns]          # 保留最近 N 轮
    旧文本 = 旧轮序列化
    新摘要 = summarize(旧文本)                 # 外部注入，测试用假函数
    turns = turns[-keep_last_turns:]          # 滑动窗口收拢
    summary = 新摘要                           # 与旧摘要的取舍：直接覆盖，
                                             # 因为旧摘要的内容已在旧文本里
    如果还超预算 → summary 截断到 summary_tokens
```

**为什么「保留最近 N 轮 + 旧轮摘要」而不是全压掉**：LLM 的回答依赖近期上下文（追问、纠正），摘要会丢细节；但全部保留会撑爆预算。分层保留是记忆管理最经典的折中：**近处给精度，远处给概要**。

### 3.3 检索与重建

```
search(query, top_k, topic):
    vec = embedder.embed_one(query)
    hits = collection.query(query_embeddings=[vec], n_results=top_k, where={...} if topic)
    # where: topic 筛选用 {"topic_tags": {"$contains": topic}}（子串匹配，ChromaDB 语法）
    return [MemoryHit(snippet=重建(hit), similarity=1 - hit.distance) for hit in hits]
```

**重建**：ChromaDB 返回的 metadata 字典键名与 ResearchSnippet 字段名对齐，可直接 `ResearchSnippet(**meta)`——claims/topic_tags 存的就是 list，原样回读；唯一的转换是 `published_at` 的 `""` → `None`（坑③：metadata 不接受 None，存时转空串）。

---

## 2.5 ChromaDB API 速查（写 vector_store 时对照）

```python
import chromadb

client = chromadb.PersistentClient(path=str(settings.chroma_path()))
# 每次启动用同一个 path → 数据持久化；不存在会自动创建目录
collection = client.get_or_create_collection(
    "snippets",
    metadata={"hnsw:space": "cosine"},   # 坑①：必须显式 cosine，默认是 L2
)

# ---- 写入（永远显式传 embeddings，坑②）----
collection.upsert(
    ids=[snippet.snippet_id],
    embeddings=[vec],                    # vec = embedder.embed_one(text)，list[float]
    metadatas=[meta],                    # 字典：值只能是 str/int/float/bool
    documents=[全文文本],                 # 可选，存一份原文方便回溯
)

# ---- 查询 ----
result = collection.query(
    query_embeddings=[vec],
    n_results=5,
    # where={"topic_tags": {"$contains": "RAG"}},   # 可选主题筛选（坑⑥）
)
# 返回结构（⚠️ 外层是 batch 维度，永远取 [0]）：
# {
#   "ids": [["id1", "id2", ...]],
#   "distances": [[0.12, 0.31, ...]],      # cosine 空间 → similarity = 1 - distance（坑④）
#   "metadatas": [[{...}, {...}]],
#   "documents": [["文本", ...]],
# }

collection.count()      # 片段总数
collection.delete(where={...})   # clear() 用：先查全部 id 再删
```

**metadata 的硬限制**（坑③最终结论，实测修正）：
- 值接受 `str / int / float / bool`，**1.x 也原生支持 `list` 值**（实测 1.5.9 通过）——claims、topic_tags **直接存 list**，省去 JSON 序列化/反序列化两套转换
- **None 仍然不行** → `published_at` 为 None 时存 `""`，重建时 `""` → `None`
- **list 值必须非空且同质**（实测 1.5.9：空 list 直接 `ValueError: Expected metadata list value ... to be non-empty`）→ 空 claims/topic_tags **不写这个键**，读回时 Pydantic 缺键用默认 `[]`（坑⑨）
- 有了 list 支持，`$contains` 就是 list 的精确成员匹配（实测中文也生效）

## 4. 设计决策与坑（写代码前先看）

| # | 决策 | 原因 |
|---|---|---|
| ① | collection 显式 `metadata={"hnsw:space": "cosine"}` | 默认 L2 空间，余弦语义全错 |
| ② | 永远显式传 embeddings，不碰默认嵌入函数 | 默认函数首次调用下载 ONNX 模型，离线挂 |
| ③ | claims/topic_tags 直接存 **list**（实测 1.x 支持）；只有 None 要转 `""` | 曾经以为 metadata 不支持 list（0.4.x 的旧限制），1.x 实测可用——省掉 JSON 序列化两套转换 |
| ④ | 相似度 = 1 - distance | 归一化向量 + cosine 空间的换算 |
| ⑤ | 写入门槛 credibility ≥ 7 在 add_snippet 内检查 | 规则集中一处，调用方不重复判断 |
| ⑥ | 主题筛选：`where={"topic_tags": {"$contains": topic}}` **下推数据库**（实测 1.5.9 生效） | 前提是 topic_tags 存 **list** 值——`$contains` 对 list 是精确成员匹配（中文也 OK）；对 JSON 字符串则永远不命中（成员 vs 子串语义错配） |
| ⑦ | compress 的 summarize 外部注入 | 短期记忆不依赖模型层，可测性 |
| ⑧ | 过期降权（§6.4）、冲突标记（§6.4）M6 不做 | 分别在阶段 2/4 落地，避免范围膨胀 |
| ⑨ | 空 list 的 claims/topic_tags **不写进 metadata** | 实测 1.5.9：list 值必须非空，空 list 抛 ValueError；缺键读回时 Pydantic 默认 `[]`，零转换（与坑③ published_at 的 `""` 哨兵同构） |
| ⑩ | manager 必须有 `add_turn`（短期记忆写入口） | §2.4 原契约漏了「写」，只有读/压；M7 每轮对话后要记录轮次，没有 add_turn 就无内容可读 |

## 5. 验收标准（我的测试会验证这些）

**vector_store（mock embedder，临时目录隔离）**：
- [ ] credibility < 7 → rejected，不入库
- [ ] 相似 snippet + claims 重合 > 80% → duplicate，不写入
- [ ] claims 重合 < 80%（观点不同）→ written，两条并存
- [ ] 空 claims → 视为不重复，写入
- [ ] 写入后可检索到，相似度符合 MockEmbedder 的信号（相关 > 无关）
- [ ] topic 筛选只返回该主题的片段
- [ ] claims/topic_tags 元数据往返：list 进 list 出，无需转换
- [ ] add_batch 返回逐条结果；count/clear 行为正确

**short_term**：
- [ ] token_count 估算正确（含摘要）
- [ ] 超预算 → should_compress True
- [ ] compress 后保留最近 N 轮 + 摘要；旧轮进了 summarize 的输入
- [ ] compress 后仍超预算 → 摘要被截断
- [ ] context 格式包含「历史摘要」与「最近对话」两段
- [ ] to_dict / from_dict 往返无损

**preferences**：
- [ ] 文件不存在 → 空偏好，不崩
- [ ] update 合并不覆盖其他键
- [ ] 写盘后可重新读回（持久化）

**manager**：
- [ ] remember / recall / context 委托正确
- [ ] mock 模式全链路离线可用

## 6. 写作顺序建议

1. **vector_store.py**（最核心，先啃硬骨头：ChromaDB 读写 + 去重）
2. **short_term.py**（滑动窗口 + 压缩）
3. **manager.py**（组合，依赖前两个）
4. **preferences.py**（最简单，最后收尾）
