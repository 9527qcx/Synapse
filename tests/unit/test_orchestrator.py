"""agents/orchestrator.py 单元测试：Reflection 循环 / 收敛降级 / 依赖顺序。

对应 M7 设计文档 §6 验收标准：
- 一轮 PASS → 全部 approved，无修订入队
- NEEDS_REVISION → 修订任务入队（source=CRITIC_REVISION），二轮通过后 approved
- 2 轮仍不过 → 降级放行（approved 含存疑片段 + contradictions 记录），不死循环
- 依赖顺序：被依赖任务先完成
- RunResult 汇总：reflection_rounds / reused_hits / errors 正确
"""
from __future__ import annotations

from synapselib.agents.orchestrator import Orchestrator
from synapselib.core.schemas import Task, TaskSource, TaskStatus
from synapselib.memory.manager import MemoryManager


def _orch(mock_settings, tools, llm):
    return Orchestrator(mock_settings, tools, MemoryManager(mock_settings), llm=llm)


def _task(desc="大模型幻觉缓解", queries=None, **over):
    return Task(description=desc, search_queries=queries or [desc], **over)


class TestPassRound:
    def test_single_pass_no_revision(self, mock_settings, make_llm, make_tools):
        """一轮 PASS：无反思、无修订任务、材料全部入库。"""
        tools = make_tools()
        r = _orch(mock_settings, tools, make_llm(verdicts=["PASS"]))
        t = _task()
        result = r.run([t])
        assert result.reflection_rounds == 0
        assert len(result.approved_snippets) == 1
        assert result.contradictions == []
        assert result.errors == []
        assert t.status == TaskStatus.COMPLETED
        assert t.reflection_count == 0
        assert all(x.source == TaskSource.INITIAL for x in result.tasks)


class TestRevisionRound:
    def test_revision_then_pass(self, mock_settings, make_llm, make_tools):
        """首轮 NEEDS_REVISION → 修订任务入队（CRITIC_REVISION）→ 二轮通过。"""
        tools = make_tools()
        r = _orch(mock_settings, tools, make_llm(verdicts=["NEEDS_REVISION", "PASS"]))
        t = _task()
        result = r.run([t])
        assert result.reflection_rounds == 1
        assert t.reflection_count == 1
        assert t.status == TaskStatus.COMPLETED
        # 修订任务存在且完成
        revisions = [x for x in result.tasks if x.source == TaskSource.CRITIC_REVISION]
        assert len(revisions) == 1
        assert revisions[0].status == TaskStatus.COMPLETED
        assert all(x.status == TaskStatus.COMPLETED for x in result.tasks)
        # 修订任务的 1 条 + 原任务二轮的 1 条
        assert len(result.approved_snippets) == 2


class TestDegrade:
    def test_two_rounds_then_degrade(self, mock_settings, make_llm, make_tools):
        """永远打回 → 2 轮后降级放行：存疑入库 + contradictions，不死循环。"""
        tools = make_tools()
        r = _orch(mock_settings, tools, make_llm(verdicts=["NEEDS_REVISION"]))
        t = _task()
        result = r.run([t])
        assert t.reflection_count == 2
        assert t.status == TaskStatus.COMPLETED, "降级后任务仍完成"
        assert len(result.contradictions) == 1
        assert "存疑" in result.contradictions[0]
        assert result.reflection_rounds == 2
        # 2 个修订任务 + 原任务：3 条材料（2 修订各 1 + 降级放行 1）
        assert len(result.approved_snippets) == 3
        assert all(x.status == TaskStatus.COMPLETED for x in result.tasks)


class TestDependency:
    def test_dependent_task_waits(self, mock_settings, make_llm, make_tools):
        """t2 依赖 t1 → 两个都完成，不因依赖死循环。"""
        tools = make_tools()
        r = _orch(mock_settings, tools, make_llm(verdicts=["PASS"]))
        t1 = _task("基础任务")
        t2 = _task("评估任务", dependencies=[t1.task_id])
        result = r.run([t1, t2])
        assert t1.status == TaskStatus.COMPLETED
        assert t2.status == TaskStatus.COMPLETED
        assert tools.search_calls == 2
        assert len(result.approved_snippets) == 2


class TestEmptyMaterial:
    def test_no_material_skips_critic(self, mock_settings, make_llm, make_tools):
        """搜索全空 → 无材料 → 直接完成，不审（critic 一次都不调）。"""
        llm = make_llm(verdicts=["PASS"])
        r = _orch(mock_settings, make_tools(empty=True), llm)
        t = _task(queries=["q1", "q2", "q3"])  # 3 个查询全空 → 触发提前终止
        result = r.run([t])
        assert t.status == TaskStatus.COMPLETED
        assert llm.critic_calls == 0
        assert result.approved_snippets == []
        assert "连续 3 次搜索无结果" in result.errors
