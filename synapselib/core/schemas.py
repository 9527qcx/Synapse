"""核心数据 Schema —— 项目大纲 §17 的代码落地。

设计约定：
- 所有枚举用 str, Enum：序列化输出字符串，拼错时 Pydantic 直接报错
- claims 存储保留原文（仅 strip），小写归一化通过 claims_normalized 属性按需取用
- 本文件不允许 import 其他业务模块，保证「数据契约」独立可复用
"""
from __future__ import annotations

import re
from enum import Enum
from typing import TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------- 枚举

class TaskStatus(str, Enum):
    """任务状态（§17.1）。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskSource(str, Enum):
    """任务来源（§17.1）：初始分解 / 批判者补充 / 重规划。"""

    INITIAL = "initial"
    CRITIC_REVISION = "critic_revision"
    REPLAN = "replan"


class SourceType(str, Enum):
    """来源类型（§17.2）。"""

    PAPER = "paper"
    BLOG = "blog"
    OFFICIAL = "official"
    NEWS = "news"


class PublicationStatus(str, Enum):
    """发表状态（§17.2）。"""

    PUBLISHED = "published"
    PREPRINT = "preprint"
    UNKNOWN = "unknown"


class Verdict(str, Enum):
    """Critic 裁决（§5.3）。"""

    PASS = "PASS"
    NEEDS_REVISION = "NEEDS_REVISION"
    CONTRADICTION = "CONTRADICTION"


# ---------------------------------------------------------------- §17.1 Task

class Task(BaseModel):
    """研究任务（§17.1）。"""

    task_id: str = Field(default_factory=lambda: uuid4().hex)
    description: str
    search_queries: list[str] = Field(default_factory=list)
    priority: int = Field(default=5, ge=1, le=10)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: str = "researcher"
    reflection_count: int = Field(default=0, ge=0)
    source: TaskSource = TaskSource.INITIAL

    @field_validator("description")
    @classmethod
    def _description_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("description 不能为空")
        return v


# ---------------------------------------------------------------- §17.2 ResearchSnippet

class ResearchSnippet(BaseModel):
    """研究片段（§17.2）—— 长期记忆的存储单元。

    - snippet_id 缺省自动生成（uuid4 hex，恰好也是 ChromaDB 的合法 id）
    - claims 逐条 strip，并过滤空串；小写归一化见 claims_normalized
    - credibility_score 越界（0-10）直接拒绝写入
    """

    snippet_id: str = Field(default_factory=lambda: uuid4().hex)
    source_url: str
    source_title: str
    source_type: SourceType
    publication_status: PublicationStatus = PublicationStatus.UNKNOWN
    claims: list[str] = Field(default_factory=list)
    evidence_excerpt: str
    credibility_score: float = Field(ge=0, le=10)
    topic_tags: list[str] = Field(default_factory=list)
    published_at: str | None = None
    extracted_at: str = Field(default_factory=lambda: _now_iso())
    task_id: str

    @field_validator("claims")
    @classmethod
    def _claims_strip(cls, v: list[str]) -> list[str]:
        cleaned = [c.strip() for c in v if c and c.strip()]
        return cleaned

    @field_validator("published_at")
    @classmethod
    def _published_at_iso(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not re.match(r"^\d{4}-\d{2}-\d{2}(T.*)?$", v):
            raise ValueError(f"published_at 必须为 ISO8601 日期，收到: {v!r}")
        return v

    @property
    def claims_normalized(self) -> set[str]:
        """小写归一化的 claims 集合，用于去重比对（§6.2 细判）。"""
        return {c.lower() for c in self.claims}


# ---------------------------------------------------------------- §17.3 CritiqueOutput

class SnippetEvaluation(BaseModel):
    """单片段审核结论（§17.3）。"""

    snippet_id: str
    credibility_score: float = Field(ge=0, le=10)
    issues: list[str] = Field(default_factory=list)
    suggestion: str = ""


class RevisionTask(BaseModel):
    """补充任务单（§17.3）—— 由 Critic 生成，直接挂入任务 DAG。"""

    task_description: str
    search_queries: list[str] = Field(default_factory=list)
    target_info: list[str] = Field(default_factory=list)
    priority: int = Field(default=5, ge=1, le=10)


class CritiqueOutput(BaseModel):
    """批判结果（§17.3）。"""

    verdict: Verdict
    overall_feedback: str = ""
    snippet_evaluations: list[SnippetEvaluation] = Field(default_factory=list)
    revision_tasks: list[RevisionTask] = Field(default_factory=list)
    conflict_details: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------- §17.4 AgentState

class AgentState(TypedDict):
    """全局状态（§17.4）—— 阶段 2 接入 LangGraph 时直接使用。"""

    research_topic: str
    tasks: list[Task]
    current_task_id: str | None
    research_snippets: list[dict]
    critique_result: dict | None
    approved_snippets: list[dict]
    final_report: str | None
    reflection_count: int
    memory_context: dict
    error_count: int
    retry_records: dict
    user_preferences: dict


# ---------------------------------------------------------------- 阶段 1 补充 Schema

class SubQueryPlan(BaseModel):
    """子查询分解输出（阶段 1 Planner 的 LLM 输出契约）。"""

    queries: list[str] = Field(min_length=1)


class SnippetDraft(BaseModel):
    """LLM 提取的片段草稿（阶段 1：不含元数据，由 Researcher 组装）。"""

    claims: list[str] = Field(default_factory=list)
    evidence_excerpt: str


class ReusedHit(BaseModel):
    """记忆复用命中（§6.4 / 架构原则 2）。"""

    snippet_id: str
    similarity: float
    source_url: str


class ResearchResult(BaseModel):
    """单任务研究结果（阶段 1 Researcher 的输出汇总）。"""

    task: Task
    snippets_written: list[ResearchSnippet] = Field(default_factory=list)
    snippets_reused: list[ReusedHit] = Field(default_factory=list)
    duplicates_skipped: int = 0
    usage: dict | None = None  # TODO(M4): 换成 models.schemas.Usage
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- 工具函数

def _now_iso() -> str:
    """本地时间 ISO8601 字符串（避免模块级依赖 datetime.now 可测性差）。"""
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")
