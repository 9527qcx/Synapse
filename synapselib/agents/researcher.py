from synapselib.core.schemas import (PublicationStatus, ReusedHit, ResearchResult,ResearchSnippet, SnippetDraft, SourceType, Task)
from synapselib.models.schemas import ChatRole, ChatMessage
from synapselib.tools.schemas import SearchResult
from synapselib.models.factory import ModelFactory
from synapselib.memory.manager import MemoryManager
from synapselib.tools.base import ToolBox
from collections.abc import Callable
from synapselib.config.settings import Settings
from synapselib.core.output_parser import parse_json             # LLM 输出解析
from synapselib.core.errors import ConfigError, OutputParseError, ToolError
import re


class Researcher:
    """执行检索任务：复用优先 → 搜索 → 提取 → 提炼 → 写入长期记忆。"""
    def __init__(self, settings: Settings, tools: ToolBox, memory: MemoryManager,
                 llm: Callable[[str, list[ChatMessage]], str] | None = None, factory: ModelFactory | None = None) -> None:
        self.settings = settings
        self.tools = tools
        self.memory = memory
        if llm is not None:
            self._llm = llm
        elif factory is not None:
            self._llm = _default_llm(factory)
        else:
            raise ConfigError("llm or factory must be provided")

    def research(self, task: Task) -> ResearchResult:
        """执行一个任务：复用 → 搜索 → 提取 → 提炼 → 写入（失败不中断，记 errors）。"""
        errors: list[str] = []
        written: list[ResearchSnippet] = []
        reused: list[ReusedHit] = []
        duplicates = rejected = 0

        # ① 复用优先：高相似记忆直接复用，不重复抓取
        threshold = self.settings.reuse_similarity_threshold
        for hit in self.memory.recall(task.description, top_k=3):
            if hit.similarity >= threshold:
                reused.append(ReusedHit(
                    snippet_id=hit.snippet.snippet_id,
                    similarity=hit.similarity,
                    source_url=hit.snippet.source_url,
                ))
        if reused: 
            return ResearchResult(task=task, snippets_reused=reused, errors=errors)
        # ② 搜索：连续 3 次无结果 → 上报终止（§5.2）
        search_results: list[SearchResult] = []
        empty_streak = 0
        for query in task.search_queries[:3]:
            resp = self.tools.search(query)
            if not resp.results:
                empty_streak += 1
                if empty_streak >= 3:
                    errors.append("连续 3 次搜索无结果")
                    return ResearchResult(task=task, errors=errors)  # 提前终止
            else:
                empty_streak = 0
                search_results.extend(resp.results)

        # ③④⑤ 提取 → 提炼 → 组装 → 写入（每步独立容错）
        for result in search_results[:3]:
            try:
                text = self.tools.extract(result.url).text      # ③ 提取
            except ToolError as e:
                errors.append(f"提取失败 {result.url}: {e}")
                continue
            try:
                draft = self._extract_draft(text)               # ④ 提炼（你写）
            except OutputParseError as e:
                errors.append(f"提炼失败 {result.url}: {e}")
                continue
            snippet = self._build_snippet(task, result, draft)  # ⑤ 组装（你写）
            r = self.memory.remember(snippet)                   # ⑤ 写入：去重/门槛在记忆层
            if r.status == "written":
                written.append(snippet)
            elif r.status == "duplicate":
                duplicates += 1
            else:
                rejected += 1

        return ResearchResult(
            task=task,
            snippets_written=written,
            snippets_reused=reused,
            duplicates_skipped=duplicates,
            snippets_rejected=rejected,
            errors=errors,
        )
    
    def _extract_draft(self, text: str) -> SnippetDraft:
        """LLM 提炼 claims + 摘录（prompt 含「提取」关键词，离线 mock 也能命中）。"""
        prompt = f"你是研究助手。请提取以下网页正文的核心主张（claims）与证据摘录：\n\n{text[:4000]}"
        response = self._llm("extract", [ChatMessage(role=ChatRole.USER, content=prompt)])
        return parse_json(response, SnippetDraft)
    
    def _build_snippet(self, task: Task, result: SearchResult, draft: SnippetDraft) -> ResearchSnippet:
        """组装研究片段：元数据来自搜索结果，可信度初评 = 来源类型分。"""
        is_arxiv = "arxiv.org" in result.url
        stype = SourceType.PAPER if is_arxiv else SourceType.BLOG
        return ResearchSnippet(
            source_url=result.url,
            source_title=result.title,
            source_type=stype,
            publication_status=PublicationStatus.PREPRINT if is_arxiv else PublicationStatus.UNKNOWN,
            claims=draft.claims,
            evidence_excerpt=draft.evidence_excerpt,
            credibility_score={  # 初评 = 来源类型分（§7.2 规则表），终评归 critic
                SourceType.PAPER: 10, SourceType.OFFICIAL: 8,
                SourceType.NEWS: 6, SourceType.BLOG: 4,
            }[stype],
            topic_tags=[task.description],   # 主题标签 = 任务描述（M6 主题筛选的钩子）
            published_at=_sanitize_date(result.published_at),  # 坑：日期必须过 ISO 校验
            task_id=task.task_id,
        )

def _default_llm(factory: ModelFactory) -> Callable[[str, list[ChatMessage]], str]:
    """默认 llm 实现：包 factory.complete，取回 content。"""
    def call(task_kind: str, messages: list[ChatMessage]) -> str:
        return factory.complete(task_kind, messages).content
    return call

def _sanitize_date(value: str | None) -> str | None:
    """只放行 ISO8601 日期（ResearchSnippet 校验器要求），其他一律 None。"""
    if value and re.match(r"^\d{4}-\d{2}-\d{2}(T.*)?$", value):
        return value
    return None





