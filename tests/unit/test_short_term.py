"""memory/short_term.py 单元测试：token 估算、压缩、上下文、持久化。

用户的三处设计改进都是验收重点（对照 §6.1 滑动窗口）：
1. 双检查守卫：轮数不够 / 未超预算 → 不调 summarize（省 LLM 调用）
2. 旧文本拼接旧摘要：新摘要覆盖历史，但旧摘要内容进入 summarize 输入，防遗忘
3. 截断取尾部：摘要超限时保留结尾（结尾离当前最近，信息最新鲜）
"""
from __future__ import annotations

import json

from synapselib.memory.short_term import ShortTermMemory


def _fake_summarize(seen: list, result: str = "S"):
    """测试专用 summarize：记录输入文本，返回固定结果。"""
    def fn(text: str) -> str:
        seen.append(text)
        return result
    return fn


class TestTokenCount:
    def test_estimate_chars_divided_by_two(self):
        """近似估算：全部字符数 // 2。"""
        m = ShortTermMemory()
        m.add("user", "abc")      # 3 chars
        m.add("assistant", "ab")  # 2 chars
        assert m.token_count() == (3 + 2) // 2

    def test_summary_counts_toward_budget(self):
        """摘要也计入预算（旧摘要 + 轮次都占 token）。"""
        m = ShortTermMemory()
        m.add("user", "x" * 10)
        m._summary = "y" * 5
        assert m.token_count() == 15 // 2


class TestCompress:
    def test_over_budget_triggers(self):
        m = ShortTermMemory(budget_tokens=10)
        m.add("user", "x" * 25)  # 25 chars → 12 tokens > 10
        assert m.should_compress() is True
        assert m.token_count() > m.budget_tokens

    def test_under_budget_no(self):
        m = ShortTermMemory(budget_tokens=10)
        m.add("user", "x" * 5)  # 2 tokens ≤ 10
        assert m.should_compress() is False

    def test_keeps_last_turns_and_summarizes_old(self):
        """压缩后：旧轮进 summarize，只剩最近 N 轮完整，摘要替换旧摘要。"""
        seen = []
        # content 不含 role 前缀：5 轮 × 5 chars = 25 chars → 12 tokens > 10
        m = ShortTermMemory(budget_tokens=10, keep_last_turns=2)
        for i in range(5):
            m.add("user", f"turn{i}")
        m.compress(_fake_summarize(seen, result="SUMMARY"))

        # 注意尾部 "\n"：old_text = 轮次拼接 + "\n" + 旧摘要（旧摘要为空）
        assert seen == ["user: turn0\nuser: turn1\nuser: turn2\n"]
        assert m._summary == "SUMMARY"
        ctx = m.context()
        assert "turn3" in ctx and "turn4" in ctx   # 最近 2 轮保留
        assert "turn0" not in ctx                  # 旧轮被摘要吸收

    def test_old_summary_appended_to_input(self):
        """改进②：第二次压缩时，旧摘要拼进 summarize 输入（防信息遗忘）。"""
        seen = []
        m = ShortTermMemory(budget_tokens=20, keep_last_turns=1)
        m.add("user", "a" * 60)   # 33 tokens > 20，且 2 轮 > 保留 1 轮
        m.add("user", "b" * 60)
        m.compress(_fake_summarize(seen, "OLD_SUMMARY"))
        seen.clear()

        m.add("user", "c" * 60)   # 再次超预算、轮数够 → 第二次压缩
        m.compress(_fake_summarize(seen, "NEW_SUMMARY"))
        assert "OLD_SUMMARY" in seen[0]

    def test_few_turns_no_summarize(self):
        """守卫①：超预算但轮数不够 → 不调 summarize（没得压）。"""
        seen = []
        m = ShortTermMemory(budget_tokens=1, keep_last_turns=10)
        for _ in range(3):
            m.add("user", "x" * 100)
        m.compress(_fake_summarize(seen))
        assert seen == []

    def test_under_budget_no_summarize(self):
        """守卫②：轮数够但没超预算 → 不调 summarize（省 LLM 调用）。"""
        seen = []
        m = ShortTermMemory(budget_tokens=10000, keep_last_turns=2)
        for i in range(5):
            m.add("user", f"turn{i}")
        m.compress(_fake_summarize(seen))
        assert seen == []
        assert m.context().count("turn") == 5  # 轮次原封不动

    def test_summary_truncated_to_budget(self):
        """压缩后仍超预算 → 摘要截断到 summary_tokens*2 字符，取尾部。"""
        seen = []
        # content 不含 role 前缀：5 轮 × 5 chars = 25 chars → 12 tokens > 10
        m = ShortTermMemory(budget_tokens=10, keep_last_turns=2, summary_tokens=3)
        for i in range(5):
            m.add("user", f"turn{i}")
        m.compress(_fake_summarize(seen, result="ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        assert m._summary == "UVWXYZ"  # max_chars = 3*2 = 6，取最后 6 字符


class TestContext:
    def test_empty_only_recent_section(self):
        m = ShortTermMemory()
        assert "【最近对话】" in m.context()
        assert "【历史摘要】" not in m.context()  # 无摘要时不出现该段

    def test_both_sections_when_summary_exists(self):
        m = ShortTermMemory()
        m.add("user", "你好")
        m._summary = "历史摘要内容"
        ctx = m.context()
        assert "【历史摘要】" in ctx and "历史摘要内容" in ctx
        assert "【最近对话】" in ctx and "user: 你好" in ctx


class TestPersistence:
    def test_dict_roundtrip_via_json(self):
        """to_dict → JSON → from_dict：状态完全一致（模拟写盘读盘）。"""
        m = ShortTermMemory(budget_tokens=1000, keep_last_turns=2)
        m.add("user", "a")
        m.add("assistant", "b")
        m._summary = "旧摘要"

        data = json.loads(json.dumps(m.to_dict()))  # 过一遍 JSON 序列化
        restored = ShortTermMemory.from_dict(data)
        assert restored.context() == m.context()
        assert restored.token_count() == m.token_count()
        # 注意：构造参数（budget 等）不属于存档状态，from_dict 不还原（契约如此）