from synapselib.tools.base import ToolBox
from synapselib.memory.manager import MemoryManager
from synapselib.config.settings import Settings
from synapselib.agents.researcher import Researcher
from synapselib.agents.critic import Critic
from synapselib.core.schemas import Task, RunResult, TaskStatus, Verdict, TaskSource



class Orchestrator:
    def __init__(self, settings: Settings, tools: ToolBox, memory: MemoryManager,
                 llm=None, factory=None) -> None:
        # 组装所有零件：Researcher(settings, tools, memory, llm) + Critic(settings, memory, llm)
        # 这就是「组合根」：所有依赖在此组装，agents 之间互不直接依赖
        self.researcher = Researcher(settings, tools, memory, llm, factory)
        self.critic = Critic(settings, memory, llm, factory)

    def run(self, tasks: list[Task]) -> RunResult:
        """执行任务列表，返回终态汇总。"""
        queue = sorted(tasks, key=lambda t: (-t.priority, t.task_id)) # 优先级高，任务ID小优先
        # 通过审核， 记忆复用命中， 未解决矛盾的描述， 	过程中的错误
        approved, reused_hits, contradictions, errors = [], [], [], []
        reflection_rounds = 0

        while queue:
            task = queue.pop(0)
            has_pending_dep = False
            for dep_id in task.dependencies:
                dep = next((t for t in tasks if t.task_id == dep_id), None)
                if dep is None or dep.status != TaskStatus.COMPLETED:
                    has_pending_dep = True
                    break
            if has_pending_dep:
                queue.append(task)
                continue
                
            task.status = TaskStatus.RUNNING
            result = self.researcher.research(task)
            reused_hits.extend(result.snippets_reused) #记忆复用命中
            errors.extend(result.errors) #过程中的错误
            if not result.snippets_written: #没有检索到任何材料
                task.status = TaskStatus.COMPLETED
                continue
            
            critique = self.critic.critique(task, result.snippets_written)

            if critique.verdict == Verdict.PASS:
                task.status = TaskStatus.COMPLETED
                approved.extend(result.snippets_written)#材料入库
            elif task.reflection_count < 2:
                task.reflection_count += 1
                reflection_rounds += 1
                for rt in critique.revision_tasks: #生成修订任务
                    rev = Task(
                        description=rt.task_description,
                        search_queries=rt.search_queries,
                        priority=rt.priority,
                        source=TaskSource.CRITIC_REVISION,
                    )
                    tasks.append(rev) #登记（依赖查找）
                    queue.append(rev) ##入队执行
                    task.dependencies.append(rev.task_id) #原任务等修订任务完成后再审
                task.status = TaskStatus.PENDING
                queue.append(task)
            else:
                task.status = TaskStatus.COMPLETED
                approved.extend(result.snippets_written)#存疑放行
                contradictions.append(f"{task.description}: 两轮反思未通过，存疑放行")

        return RunResult(
            topic=tasks[0].description,
            tasks=tasks,
            approved_snippets=approved,
            reused_hits=reused_hits,
            contradictions=contradictions,
            errors=errors,
            reflection_rounds=reflection_rounds,
        )
            

                



