"""全局状态辅助 —— AgentState（§17.4）的构造与初始化。

阶段 1 不使用 LangGraph，但状态契约按 §17.4 立住；
阶段 2 接入 LangGraph 时直接复用同一份 TypedDict。
"""
from __future__ import annotations

from .schemas import AgentState


def empty_state(topic: str) -> AgentState:
    """构造一个全字段为空的 AgentState（阶段 1 只使用部分字段）。"""
    return {
        "research_topic": topic,
        "tasks": [],
        "current_task_id": None,
        "research_snippets": [],
        "critique_result": None,
        "approved_snippets": [],
        "final_report": None,
        "reflection_count": 0,
        "memory_context": {},
        "error_count": 0,
        "retry_records": {},
        "user_preferences": {},
    }
