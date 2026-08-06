"""工具层公共设施：指数退避重试装饰器 + ToolBox 聚合容器。

设计要点（呼应 M4 模型层的「降级链」思路）：
- 模型层失败 → 换一个 provider（配置问题，降级）
- 工具层失败 → 同一工具重试（网络波动，等等就好）
  两个模式都是「抛出失败信号 + 上层兜底」，只是兜底策略不同。

重试语义（这是今天最重要的知识点）：
- 只重试 ToolError：网络超时、限流、解析失败 —— 重试有意义
- 不重试 ConfigError / 程序 bug：重试一万次结果一样，只会放大问题
  所以 ConfigError 甚至不该被本装饰器看见 —— 它在异常继承树上
  （SynapseError → ToolError / ConfigError）是平行的，天然不匹配
- mock 模式工具不抛 ToolError，装饰器对 mock 是无操作路径
"""
from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from synapselib.core.errors import RetryExhaustedError, ToolError
from synapselib.observability.tracing import record_event

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    *,
    exceptions: Sequence[type[Exception]] = (ToolError,),
) -> Callable[[F], F]:
    """指数退避重试装饰器（大纲 §13.5：每个工具封装内实现指数退避重试）。

    语义：
    - 第 1 次失败 → sleep base_delay 后重试
    - 第 2 次失败 → sleep base_delay * 2 …… 以此类推（指数退避）
    - 达到 max_retries 次重试仍失败 → 抛 RetryExhaustedError
      （继承 ToolError，调用方只需按工具失败处理）
    - 只捕获 exceptions 指定的异常（默认 ToolError），其他异常直接穿透

    参数：
        max_retries: 最多重试几次（不含首次调用），默认 3
        base_delay:  首次重试前的等待秒数，之后翻倍
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            last_error: Exception | None = None
            # attempt = 已重试次数：0 时是首次调用，>0 时是重试
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:  # type: ignore[arg-type]
                    last_error = e
                    if attempt >= max_retries:
                        break  # 重试次数已用完，跳出循环抛异常
                    record_event(
                        "tool_retry",
                        tool=fn.__name__,
                        attempt=attempt + 1,
                        delay_ms=round(delay * 1000),
                    )
                    logger.warning(
                        "[%s] 第 %d 次失败：%s，%.1fs 后重试", fn.__name__, attempt + 1, e, delay
                    )
                    time.sleep(delay)
                    delay *= 2  # 指数退避
            raise RetryExhaustedError(
                f"{fn.__name__} 重试 {max_retries} 次后仍失败：{last_error}"
            ) from last_error

        return wrapper  # type: ignore[return-value]

    return decorator


class ToolBox:
    """工具聚合容器 —— M7 Researcher 访问工具的**唯一入口**。

    为什么需要它（而不是 Researcher 直接 new 三个工具）：
    1. 依赖注入集中点：Settings 只在这里读取一次，工具本身不碰全局配置
    2. 模式一致性：mock_mode 在这里统一透传，Researcher 不需要知道
       每个工具各自怎么处理 mock
    3. 测试友好：换工具实现（比如 SerpAPI 替换 SearXNG）只动这一处
    """

    def __init__(self, settings: Any) -> None:
        from synapselib.tools.arxiv_tool import ArxivTool
        from synapselib.tools.searxng import SearXNGClient
        from synapselib.tools.web_extractor import WebExtractor

        self.searxng = SearXNGClient(
            base_url=settings.search_base_url,
            mock_mode=settings.mock_mode,
        )
        self.arxiv = ArxivTool(mock_mode=settings.mock_mode)
        self.extractor = WebExtractor(mock_mode=settings.mock_mode)

        self._tools = {"search": self.searxng, "papers": self.arxiv, "extract": self.extractor}

    def get(self, name: str) -> Any:
        """按名字取工具（权限控制点：M7 只开放这 3 个名字，见大纲 §11 工具权限）。"""
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(f"未知工具：{name!r}，可用：{sorted(self._tools)}") from None

    # ---- 便捷方法：M7 代码不用再写 toolbox.get("search").search(...) 这种长链 ----

    def search(self, query: str, max_results: int = 5) -> Any:
        """网页搜索（SearXNG / mock）。"""
        return self.searxng.search(query, max_results=max_results)

    def fetch_papers(self, query: str, max_results: int = 5) -> Any:
        """ArXiv 论文检索（SDK / mock）。"""
        return self.arxiv.search(query, max_results=max_results)

    def extract(self, url: str) -> Any:
        """网页正文提取（Trafilatura / mock）。"""
        return self.extractor.extract(url)
