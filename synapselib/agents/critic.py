from synapselib.memory.manager import MemoryManager
from synapselib.config.settings import Settings
from synapselib.models.factory import ModelFactory
from synapselib.core.errors import ConfigError, OutputParseError
from synapselib.core.schemas import Task, CritiqueOutput, ResearchSnippet, SourceType, SnippetEvaluation, Verdict, RevisionTask
from synapselib.models.schemas import ChatMessage, ChatRole
from synapselib.core.output_parser import parse_json
from collections.abc import Callable
from datetime import datetime



class Critic:
    def __init__(self, settings: Settings, memory: MemoryManager, llm=None, factory: ModelFactory | None = None) -> None:
        # llm 与 researcher 同款注入（缺省包 factory？不——Critic 构造契约里只有 llm）
        # 但为了统一，也接受 factory，规则同 Researcher：llm 优先，其次 factory，都没有 → ConfigError
        self.memory = memory
        self.settings = settings

        if llm is not None:
            self._llm = llm
        elif factory is not None:
            self._llm = _default_llm(factory)
        else:
            raise ConfigError("Critic 构造时必须提供 llm 或 factory 参数")

    
    def critique(self, task: Task, snippets: list[ResearchSnippet]) -> CritiqueOutput:
        """对任务进行批判性分析，返回分析结果。task任务，snippets研究片段列表由researcher提供。"""
        counts = {}
        for i, s1 in enumerate(snippets):
            support = 1
            for j,s2 in enumerate(snippets):
                if i == j: 
                    continue
                if s1.source_url == s2.source_url:# 同一来源
                    continue
                if s1.claims_normalized & s2.claims_normalized:# 有重叠的断言
                    support += 1
            counts[s1.snippet_id] = support
        # 从记忆中召回与任务相关的研究片段
        history = self.memory.recall(task.description, top_k=3)
        prompt = f"""你是研究审核员。请审核以下研究片段，判断它们是否足以回答研究任务。
                  研究任务：{task.description}
                  历史记忆（检查新片段与既有记忆是否冲突）：
                  {[f"来源: {h.snippet.source_url} | 主张: {h.snippet.claims}" for h in history]}
                  本次研究片段：
                  {[f"编号: {s.snippet_id} | 来源: {s.source_url} 标题: {s.source_title} 类型: {s.source_type.value} | 主张: {s.claims} | 摘录: {s.evidence_excerpt} | 初评: {s.credibility_score} | N 个来源支持: {counts[s.snippet_id]}" for s in snippets]}

                  请以 JSON 输出审核结果，字段：
                  - verdict: "PASS" 或 "NEEDS_REVISION"
                  - overall_feedback: 总评
                  - snippet_evaluations: [{{"snippet_id", "credibility_score", "issues": [], "suggestion"}}]
                  - revision_tasks: [{{"task_description", "search_queries", "target_info", "priority"}}]
                  - conflict_details: 与历史记忆的冲突描述列表
                  """
        response = self._llm("critic", [ChatMessage(role=ChatRole.USER, content=prompt)])
        try:
            output = parse_json(response, CritiqueOutput)
        except OutputParseError:
            raise OutputParseError(
                response,
                "Critic output is not valid JSON. Please check the format."
            )
        
        lowest = 100
        # 把 LLM 给的评估建成查找表 evals是snippet_id到评估的映射 
        evals = {e.snippet_id: e for e in output.snippet_evaluations}
        for s in snippets:
            score = score_credibility(s.source_type, s.published_at, counts[s.snippet_id])
            if s.snippet_id in evals:
                evals[s.snippet_id].credibility_score = score #覆盖LLM的分数
            else:#LLM漏评, 防线补充一个评估
                output.snippet_evaluations.append(
                    SnippetEvaluation(
                        snippet_id=s.snippet_id,
                        credibility_score=score,
                        issues=[],
                        suggestion=""
                    )
                )
            lowest = min(lowest, score)

        if lowest < self.settings.credibility_score_threshold:
            output.verdict = Verdict.NEEDS_REVISION
            if not output.revision_tasks:
                output.revision_tasks.append(RevisionTask(
                    task_description=f"补充 {task.description} 的权威来源",
                    search_queries=[task.description],
                    target_info=["权威论文或官方文档"],
                    priority=7
                ))
        return output



def _default_llm(factory: ModelFactory) -> Callable[[str, list[ChatMessage]], str]:
    """默认 llm 实现：包 factory.complete，取回 content。"""
    def call(task_kind: str, messages: list[ChatMessage]) -> str:
        return factory.complete(task_kind, messages).content
    return call

def score_credibility(source_type: SourceType, published_at: str | None, corroborations: int=1) -> float:
    """根据研究片段的来源类型、发布时间、验证数量，计算其可信度分数。"""
    SOURCE_SCORES = {SourceType.PAPER: 10, SourceType.OFFICIAL: 8,
                     SourceType.NEWS: 6, SourceType.BLOG: 4}

    source_score = SOURCE_SCORES[source_type]
    publish_score = 0
    if published_at is None:
        publish_score = 5
    else:
        try:
            date = datetime.fromisoformat(published_at)
        except ValueError:
            publish_score = 5
        else:
            years = (datetime.now() - date).days / 365
            if years <= 2:
                publish_score = 10
            elif years <= 3:
                publish_score = 7
            else:
                publish_score = max(0, 7 - int(years - 3) * 2)

    corroborations_score = min(30, corroborations*10)
    return round(min(10.0,source_score*0.6 + publish_score*0.1 + corroborations_score*0.3), 2)





        