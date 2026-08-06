"""SearXNG 网页搜索 —— 大纲 §9「本地优先」方案（Docker 部署，零成本无限制）。

SearXNG 是一个自托管的元搜索引擎，提供 JSON API：
    GET {base_url}/search?q=<查询词>&format=json
返回 {"results": [{"title": ..., "url": ..., "content": ..., "publishedDate": ...}]}

设计：
- DI：base_url / timeout 由构造参数注入，不读全局配置（M4 同款原则）
- 真实模式：连接失败 / 非 2xx → ToolError（交给重试装饰器）
- mock 模式：直接返回 mock_data，完全不碰网络（装饰器的无操作路径）
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from synapselib.core.errors import ToolError
from synapselib.tools.base import retry_with_backoff
from synapselib.tools.mock_data import mock_search_results
from synapselib.tools.schemas import SearchResponse, SearchResult

logger = logging.getLogger(__name__)


class SearXNGClient:
    """本地 SearXNG 搜索客户端。"""

    name = "search"  # ToolBox 注册名（大纲 §11 工具权限）

    def __init__(self, base_url: str, timeout: float = 10.0, mock_mode: bool = False) -> None:
        self._base_url = base_url.rstrip("/")
        self._mock = mock_mode
        # 复用同一连接池（keep-alive），多次搜索不必重复握手
        self._client = httpx.Client(timeout=timeout)

    # 重试装饰器挂在公开方法上：真实模式网络失败会指数退避重试，
    # mock 模式永不抛 ToolError，装饰器直接放行 —— 两个路径共享一份调用语义。
    @retry_with_backoff(max_retries=2, base_delay=0.5)
    def search(self, query: str, max_results: int = 5) -> SearchResponse:
        if self._mock:
            return SearchResponse(query=query, results=mock_search_results(max_results))

        try:
            resp = self._client.get(
                f"{self._base_url}/search",
                params={"q": query, "format": "json"},
                headers={"User-Agent": "Synapse/0.1 (research assistant)"},
            )
            resp.raise_for_status()  # 非 2xx → HTTPStatusError
            payload: dict[str, Any] = resp.json()
        except httpx.RequestError as e:
            raise ToolError(f"SearXNG 连接失败（{self._base_url}）：{e}") from e
        except httpx.HTTPStatusError as e:
            raise ToolError(f"SearXNG 返回 HTTP {e.response.status_code}") from e

        # 逐条容错：SearXNG 条目字段可能缺失，缺啥补默认值，不让单条坏数据毁掉整批
        results = [
            SearchResult(
                title=item.get("title", "") or "(无标题)",
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                published_at=item.get("publishedDate"),
            )
            for item in payload.get("results", [])
        ][:max_results]
        logger.info("SearXNG 搜索 %r：命中 %d 条，返回 %d 条", query, len(results), min(len(results), max_results))
        return SearchResponse(query=query, results=results)

    def close(self) -> None:
        """关闭连接池（进程退出前调用；不关也只是进程回收）。"""
        self._client.close()
