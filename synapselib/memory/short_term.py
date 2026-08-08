"""短期记忆（会话级）：对话轮次 + 摘要压缩 —— 大纲 §6.1。

- 装什么：会话内对话轮次（_turns）+ 历史摘要（_summary）
- 压缩策略：超过 token 预算时，旧轮（除最近 N 轮）压成摘要；
  summarize 由外部注入（测试传假函数，M7 传模型层），本类不依赖 LLM
- 生命周期：一次会话内；会话结束的「结论沉淀」是 vector_store 的职责
- 持久化：to_dict / from_dict 只做「对象 ↔ dict」转换，读写盘由调用方负责
"""
from collections.abc import Callable


class ShortTermMemory:
    def __init__(self, budget_tokens: int = 8000, keep_last_turns: int = 6, summary_tokens: int = 2000) -> None:
        self._turns: list[dict] = []   # [{"role": ..., "content": ...}]
        self._summary: str = ""        # 历史摘要
        self.budget_tokens = budget_tokens
        self.keep_last_turns = keep_last_turns
        self.summary_tokens = summary_tokens

    def add(self, role: str, content: str) -> None:
        """追加一轮对话（不触发压缩：压缩是「读前自查」，见 summarize_if_needed）。"""
        self._turns.append({"role": role, "content": content})

    def token_count(self) -> int:
        """近似 token 数：全部字符数 ÷ 2（中文约 1.5~2 字符/token）。"""
        chars = sum(len(t["content"]) for t in self._turns) + len(self._summary)
        return chars // 2
    
    def should_compress(self) -> bool:
        """是否需要压缩（超预算 → True）。"""
        return self.token_count() > self.budget_tokens
    
    def compress(self, summarize: Callable[[str], str]) -> None: 
        """压缩历史对话：保留最近 N 轮完整对话，摘要为新摘要。"""
        if len(self._turns) <= self.keep_last_turns:
            return
        if not self.should_compress():
            return
        old = self._turns[:-self.keep_last_turns]
        self._turns = self._turns[-self.keep_last_turns:]

        old_text = "\n".join(f"{t['role']}: {t['content']}" for t in old) + "\n" + self._summary
        self._summary = summarize(old_text)

        max_chars = self.summary_tokens * 2
        if len(self._summary) > max_chars:
            self._summary = self._summary[-max_chars:]

    def context(self) -> str:                     # "【历史摘要】…【最近对话】…"
        """组装「历史摘要 + 最近对话」，M7 拼进提示词用。"""
        parts = []
        if self._summary:
            parts.append(f"【历史摘要】\n{self._summary}")

        recent = "\n".join(f"{t['role']}: {t['content']}" for t in self._turns)
        parts.append(f"【最近对话】\n{recent}")
        return "\n\n".join(parts)
    
    def to_dict(self) -> dict:
        """对象状态 → 可序列化 dict（JSON 可直接 dump，存哪由调用方决定）。"""
        return {"turns": self._turns, "summary": self._summary}

    @classmethod
    def from_dict(cls, data: dict) -> "ShortTermMemory":
        """从存档 dict 还原对象（替代构造函数：从存档创建，而非先建空对象）。"""
        m = cls()  # 等价于 ShortTermMemory()
        m._turns = data["turns"]
        m._summary = data["summary"]
        return m
