"""Mock 工具数据：离线、确定性 —— 大纲 §9「支持 mock 模式，方便无网调试」。

原则（与 MockProvider 一脉相承）：
- 固定内容 + 固定顺序：同一次调用永远返回相同结果，测试才可预期
- 内容围绕验收主题 T1「大模型幻觉缓解技术」编写，方便 M9 验收时对照
- 论文两条用的是真实存在的 arXiv 论文（元数据真实，只是检索是假的）
"""
from __future__ import annotations

from synapselib.tools.schemas import ExtractionResult, PaperMeta, SearchResult

# ---------------------------------------------------------------- 网页搜索

MOCK_SEARCH_RESULTS: list[SearchResult] = [
    SearchResult(
        title="大模型幻觉问题与缓解技术综述（Mock 示例）",
        url="https://arxiv.org/abs/2312.10997",
        snippet="综述了幻觉的定义、成因（数据偏差、训练目标缺陷）与主流缓解手段："
        "检索增强、事实核查、自我一致性等。",
        published_at="2025-01-15",
    ),
    SearchResult(
        title="检索增强生成如何降低大模型幻觉（Mock 示例）",
        url="https://example.com/rag-against-hallucination",
        snippet="RAG 通过把生成过程锚定到外部知识，显著降低开放领域问答的幻觉率。",
    ),
    SearchResult(
        title="大模型幻觉的评测基准与指标（Mock 示例）",
        url="https://example.com/hallucination-benchmarks",
        snippet="介绍了 HaluEval、TruthfulQA 等基准以及事实性（factuality）指标。",
        published_at="2024-11-02",
    ),
]

MOCK_PAPERS: list[PaperMeta] = [
    PaperMeta(
        paper_id="2312.10997",
        title="Retrieval-Augmented Generation for Large Language Models: A Survey",
        abstract="大规模语言模型在知识密集任务中表现出色，但面临幻觉与知识过时问题。"
        "本综述系统梳理了 RAG 的范式（预训练/微调/推理阶段）、技术组成与未来方向。",
        url="https://arxiv.org/abs/2312.10997",
        authors=["Yunfan Gao", "Yun Xiong", "Xinyu Gao", "Kangxiang Jia", "Jinliu Pan",
                 "Yuxi Bi", "Yi Dai", "Jiawei Sun", "Haofen Wang"],
        published_at="2023-12-18",
    ),
    PaperMeta(
        paper_id="2102.09736",
        title="Survey of Hallucination in Natural Language Generation",
        abstract="自然语言生成中的幻觉问题综述：定义幻觉的类型、成因，"
        "以及面向数据/模型/解码等环节的缓解策略与评估方法。",
        url="https://arxiv.org/abs/2102.09736",
        authors=["Ziwei Ji", "Nayeon Lee", "Rita Frieske", "Tiezheng Yu", "Dan Su",
                 "Yan Xu", "Etsuko Ishii", "Ye Jin Bang", "Andrea Madotto", "Pascale Fung"],
        published_at="2021-02-19",
    ),
]

# ---------------------------------------------------------------- 正文提取

MOCK_EXTRACTED_TEXT: str = """# 大模型幻觉：成因与缓解实践（Mock 正文示例）

## 幻觉从哪来

幻觉的本质是模型生成的内容与训练数据或客观事实不一致。三个主要成因：
第一，训练数据本身含有错误或偏见信息，模型把噪音当成了规律；
第二，生成机制天然倾向于"看起来合理"的续写，而不是"事实正确"的陈述；
第三，知识截止日期之后的新事实，模型完全无法感知。

## 缓解手段的工程实践

检索增强生成（RAG）是目前落地最广的方案：把问题先在外部知识库中检索，
把检索到的证据拼进提示词，再让模型基于证据回答。实证表明，
RAG 能显著降低开放领域问答的幻觉率，尤其在"知识密集、事实可查"的场景。

另一种思路是事后核查：模型先给出答案，再用检索或交叉验证的方式
判断答案中的事实性主张是否有证据支撑，没有支撑的主张予以修正或拒答。

## 评估与权衡

幻觉评测通常用 TruthfulQA 等基准，指标包括事实性、准确率与忠实度。
值得注意的是，缓解幻觉常常伴随生成"更保守"（更倾向拒答）的副作用，
生产系统需要在覆盖率与正确率之间做显式取舍。
"""


def mock_search_results(max_results: int) -> list[SearchResult]:
    """按数量截断搜索假数据（保持确定性，只截断不随机）。"""
    return MOCK_SEARCH_RESULTS[:max_results]


def mock_papers(max_results: int) -> list[PaperMeta]:
    """按数量截断论文假数据。"""
    return MOCK_PAPERS[:max_results]


def mock_extraction(url: str) -> ExtractionResult:
    """固定正文假提取（title 从 URL 猜一个，模拟真实行为）。"""
    return ExtractionResult(
        url=url,
        title="大模型幻觉：成因与缓解实践（Mock）",
        text=MOCK_EXTRACTED_TEXT,
        truncated=False,
    )
