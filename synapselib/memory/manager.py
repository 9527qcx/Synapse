from collections.abc import Callable
from synapselib.core.schemas import ResearchSnippet
from synapselib.memory.vector_store import AddResult, VectorStore, MemoryHit
from synapselib.memory.short_term import ShortTermMemory
from synapselib.config.settings import Settings
from synapselib.models.embeddings import get_embedder
from synapselib.memory.preferences import UserPreferences


class MemoryManager:
    """记忆系统唯一入口。M7 Researcher 只 import 它。"""
    def __init__(self, settings: Settings, embedder=None) -> None: 
        self.settings = settings
        self.embedder = embedder or get_embedder(settings)
        self.vector_store = VectorStore(settings, self.embedder)
        self.stm = ShortTermMemory(
            budget_tokens=settings.stm_budget_tokens,
            keep_last_turns=settings.stm_keep_last_turns,
        )
        self._prefs = UserPreferences(settings.preferences_path())
    def remember(self, snippet: ResearchSnippet) -> AddResult:   
        """将研究片段写入记忆系统。"""
        return self.vector_store.add_snippet(snippet)
    
    def recall(self, query: str, top_k: int = 5, topic: str | None = None) -> list[MemoryHit]: 
        """根据查询召回记忆片段。"""
        return self.vector_store.search(query, top_k, topic)
    
    def context(self) -> str:                                     # 短期记忆上下文
        """生成短期记忆上下文。"""
        return self.stm.context()

    def summarize_if_needed(self, summarize: Callable[[str], str]) -> None: 
        """如果需要，总结短期记忆。"""
        self.stm.compress(summarize)

    def preferences(self) -> UserPreferences: 
        """返回用户偏好。"""
        return self._prefs
    def add_turn(self, role: str, content: str) -> None: 
        """将对话轮次写入短期记忆。"""
        self.stm.add(role, content)
