"""工具层输入输出 Schema —— 大纲 §9「工具封装原则」：全部使用 Pydantic 定义。

与 core/schemas.py 的分工：
- core/schemas.py   = 领域数据契约（Task、ResearchSnippet……），整个系统共享
- tools/schemas.py  = 工具层的「对外接口契约」，描述一次工具调用会收到什么

先定义契约再写实现的好处：工具实现、M7 Researcher、测试三方都对着同一个
形状编程，谁都不会猜字段名。
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SearchResult(BaseModel):
    """单条网页搜索结果（SearXNG 的 results[] 条目）。"""

    title: str
    url: str
    snippet: str = ""  # SearXNG 的 content 字段（搜索摘要）
    published_at: str | None = None  # 发布/更新时间，缺省时允许为空

    @field_validator("title")
    @classmethod
    def _title_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("title 不能为空")
        return v


class SearchResponse(BaseModel):
    """搜索响应：query 原样回显（方便审计「这条结果是搜什么搜出来的」）+ 结果列表。"""

    query: str
    results: list[SearchResult] = Field(default_factory=list)


class PaperMeta(BaseModel):
    """论文元数据（ArXiv 检索的返回，后续直接转 ResearchSnippet 用）。

    只保留 Researcher 组装 snippet 需要的字段，摘要当 evidence 素材。
    """

    paper_id: str  # arxiv 编号，如 "2312.10997"
    title: str
    abstract: str = ""
    url: str
    authors: list[str] = Field(default_factory=list)
    published_at: str | None = None  # "YYYY-MM-DD"，兼容 ResearchSnippet.published_at 的 ISO 校验


class ExtractionResult(BaseModel):
    """网页正文提取结果：URL 原样回显 + 纯文本正文。

    truncated=True 表示正文超过 max_chars 被截断（长文论文不拦腰截断
    snippet 组装逻辑也能感知，不会把半句话当 claims 素材）。
    """

    url: str
    title: str = ""
    text: str = ""
    truncated: bool = False
