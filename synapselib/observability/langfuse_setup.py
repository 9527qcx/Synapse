"""LangFuse 守卫式初始化。

启用条件（settings.langfuse_enabled）：public_key 与 secret_key 均非空。
任何失败（缺 key / 导入失败 / 客户端创建失败）→ 降级 no-op，绝不阻塞业务。
"""
from __future__ import annotations

import atexit
import logging

from synapselib.config.settings import Settings

from . import tracing

logger = logging.getLogger(__name__)


def init_langfuse(settings: Settings) -> None:
    """初始化 LangFuse 追踪。必须在业务代码执行前调用一次。"""
    if not settings.langfuse_enabled:
        tracing.set_tracer(None)
        logger.info("LangFuse 未启用（缺少 public/secret key），追踪为 no-op")
        return

    try:
        from langfuse import Langfuse  # 延迟导入：未安装也不阻塞

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            flush_at=10,
        )
        tracing.set_tracer(client, host=settings.langfuse_host)
        atexit.register(client.flush)  # 进程退出前冲刷未发送的数据
        logger.info("LangFuse 已启用: %s", settings.langfuse_host)
    except Exception as e:  # noqa: BLE001 守卫：初始化失败不影响业务
        tracing.set_tracer(None)
        logger.warning("LangFuse 初始化失败，降级为 no-op: %s", e)
