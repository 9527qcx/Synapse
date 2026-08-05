"""模型层 IO Schema：所有 LLM 调用的输入输出契约。

与 core/schemas.py 的分工：本文件只管「模型的通信格式」，
领域数据（Task/ResearchSnippet 等）一律在 core/schemas.py。
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ChatRole(str, Enum):
    """对话角色。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """一条对话消息。"""

    role: ChatRole
    content: str


class Usage(BaseModel):
    """Token 消耗统计。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelResponse(BaseModel):
    """一次完整调用的返回。

    - model: 实际落地的模型名（降级时可能与配置名不同，可观测的关键字段）
    - provider: deepseek / ollama / mock
    - usage: 可能缺失（如 MockProvider 不模拟消耗）
    """

    content: str
    model: str
    provider: str
    usage: Usage | None = None
    latency_ms: float = Field(default=0.0, ge=0)
