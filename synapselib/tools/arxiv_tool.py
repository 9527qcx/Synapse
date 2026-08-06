"""ArXiv 论文检索 —— 大纲 §9「完全免费」的学术论文数据源。

用 arxiv 官方 Python SDK（requests 封装）。核心用法：
    search = arxiv.Search(query="关键词", max_results=N, sort_by=arxiv.SortCriterion.Relevance)
    client = arxiv.Client()
    results = list(client.results(search))   # 惰性迭代器，list() 才真正发请求

关键设计（异常分层，呼应 tools/base.py 的文档）：
- arxiv SDK 未安装 → ConfigError：属于环境配置问题，重试不会解决，直接抛
- 网络失败 / SDK 内部异常 → ToolError：重试有意义，交给重试装饰器
"""
from __future__ import annotations

import logging
from typing import Any

from synapselib.core.errors import ConfigError, ToolError
from synapselib.tools.base import retry_with_backoff
from synapselib.tools.mock_data import mock_papers
from synapselib.tools.schemas import PaperMeta

logger = logging.getLogger(__name__)


class ArxivTool:
    """ArXiv 论文检索工具。"""

    name = "papers"  # ToolBox 注册名

    def __init__(self, mock_mode: bool = False, timeout: float = 30.0) -> None:
        self._mock = mock_mode
        self._timeout = timeout

    @retry_with_backoff(max_retries=2, base_delay=0.5)
    def search(self, query: str, max_results: int = 5) -> list[PaperMeta]:
        if self._mock:
            return mock_papers(max_results)
        return self._search_real(query, max_results)

    def _search_real(self, query: str, max_results: int) -> list[PaperMeta]:
        try:
            import arxiv  # 延迟导入：mock 模式不加载 SDK
        except ImportError as e:
            raise ConfigError("arxiv SDK 未安装，请先 pip install arxiv（清华镜像）") from e

        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            results = list(arxiv.Client().results(search))
        except Exception as e:  # noqa: BLE001
            # SDK 的网络异常类型杂（URLError / 超时 / 空页……），
            # 工具层统一收窄为 ToolError 是合理取舍 —— 重试语义由装饰器保证
            raise ToolError(f"arxiv 检索失败：{e}") from e

        papers = [self._to_paper_meta(r) for r in results]
        logger.info("arxiv 检索 %r：命中 %d 篇", query, len(papers))
        return papers

    @staticmethod
    def _to_paper_meta(r: Any) -> PaperMeta:  # noqa: ANN401
        """arxiv 结果对象 → PaperMeta。

        两个常见的脏数据处理：
        - title 含换行（SDK 原始返回的排版问题）→ 折叠为单空格
        - 新版 SDK 会在标题末尾追加防误用标记 "(arXiv:xxxx)" → 剥掉
        """
        raw_title = " ".join(r.title.split())
        title = raw_title.split("(arXiv:")[0].strip() or raw_title
        paper_id = r.get_short_id().split("v")[0]  # "2312.10997v1" → "2312.10997"
        return PaperMeta(
            paper_id=paper_id,
            title=title,
            abstract=r.summary.strip(),
            url=f"https://arxiv.org/abs/{paper_id}",
            authors=[a.name for a in r.authors],
            published_at=str(r.published.date()),  # date → "YYYY-MM-DD"
        )
