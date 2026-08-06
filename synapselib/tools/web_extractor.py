"""网页正文提取 —— 大纲 §9「Trafilatura + requests，过滤广告与导航」。

Trafilatura 是本项目的核心内容获取工具：给定 HTML，它基于 DOM 结构
识别正文区（跳过导航、广告、评论区），输出纯文本 —— 比正则剥标签靠谱得多。

管线三步：
    1. httpx.get(url) 拿 HTML（带浏览器 UA，部分站点拒绝无 UA 请求）
    2. 正则抠 <title>（Trafilatura 的正文输出不含标题）
    3. trafilatura.extract(html) 提正文，超过 max_chars 截断并标记

异常分层与前两个工具一致：SDK 未装 → ConfigError；网络/解析失败 → ToolError。
"""
from __future__ import annotations

import logging
import re

import httpx

from synapselib.core.errors import ConfigError, ToolError
from synapselib.tools.base import retry_with_backoff
from synapselib.tools.mock_data import mock_extraction
from synapselib.tools.schemas import ExtractionResult

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"


class WebExtractor:
    """网页正文提取工具。"""

    name = "extract"  # ToolBox 注册名

    def __init__(self, mock_mode: bool = False, timeout: float = 15.0, max_chars: int = 8000) -> None:
        self._mock = mock_mode
        self._timeout = timeout
        self._max_chars = max_chars

    @retry_with_backoff(max_retries=2, base_delay=0.5)
    def extract(self, url: str) -> ExtractionResult:
        if self._mock:
            return mock_extraction(url)
        return self._extract_real(url)

    def _extract_real(self, url: str) -> ExtractionResult:
        try:
            import trafilatura  # 延迟导入：mock 模式不加载
        except ImportError as e:
            raise ConfigError("trafilatura 未安装，请先 pip install trafilatura（清华镜像）") from e

        html, title = self._download(url)

        try:
            text = trafilatura.extract(
                html,
                include_links=False,
                include_images=False,
                include_comments=False,
            )
        except Exception as e:  # noqa: BLE001  解析失败 → 可重试的工具错误
            raise ToolError(f"正文解析失败：{e}") from e

        if not text:  # 页面没有可识别的正文区（JS 渲染站点的典型症状）
            raise ToolError(f"未能从 {url} 提取到正文（可能是 JS 渲染页面）")

        text = text.strip()
        truncated = len(text) > self._max_chars
        if truncated:
            # 截在段落边界更安全？不 —— 保持简单，硬截断并让 truncated 标志说明一切
            text = text[: self._max_chars]
        logger.info("提取 %s：%d 字符%s", url, len(text), "（已截断）" if truncated else "")
        return ExtractionResult(url=url, title=title, text=text, truncated=truncated)

    def _download(self, url: str) -> tuple[str, str]:
        """下载 HTML + 提取标题。网络类异常在这里统一映射为 ToolError。"""
        try:
            resp = httpx.get(
                url,
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": _UA},
            )
            resp.raise_for_status()
        except httpx.RequestError as e:
            raise ToolError(f"下载 {url} 失败：{e}") from e
        except httpx.HTTPStatusError as e:
            raise ToolError(f"下载 {url} 返回 HTTP {e.response.status_code}") from e

        html = resp.text
        # 简单可靠的标题提取：<title> 里再剥掉可能残留的标签
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else ""
        return html, title
