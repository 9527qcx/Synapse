# M7 多智能体协同设计（规划稿）

> 依据大纲 §5（角色设计）、§7（Reflection 机制）、§8（协调器）。
> M7 落地「阶段 2 + 阶段 3 核心」：Researcher → Critic → 修订循环闭环。
> Synthesizer（报告生成）与 demo 一起放 M8。

## 1. 模块结构与职责

```
synapselib/agents/
├── __init__.py        # 包说明
├── researcher.py      # 研究员（§5.2）：记忆复用 → 搜索 → 提取 → 提炼 claims → 写入
├── critic.py          # 批判者（§5.3）：LLM 审核 + 确定性评分 + 修订任务单
└── orchestrator.py    # 协调器（§5.5/§8）：任务队列 + Reflection 循环 + 收敛降级
```

**依赖（已就绪，直接用）**：
- `core/schemas.py`：`Task`/`ResearchSnippet`/`SnippetDraft`/`CritiqueOutput`/`SnippetEvaluation`/`RevisionTask`/`ReusedHit`/`ResearchResult`/`TaskStatus`/`Verdict`/`TaskSource`
- `core/output_parser.py`：`parse_json(content, schema)` → 从 LLM 输出提取 JSON 并校验（失败抛 `OutputParseError`）
- `models/factory.py`：`ModelFactory.complete(task_kind, messages, **kwargs)` → 路由 + 降级链（M4 已验收）
- `tools/base.py`：`ToolBox.search / fetch_papers / extract`（mock 模式离线，M5 已验收）
- `memory/manager.py`：`MemoryManager.remember / recall`（M6 已验收）

## 2. 设计哲学：确定性 vs LLM 的分工（核心教学点）

| 逻辑 | 用什么实现 | 为什么 |
|---|---|---|
| 可信度评分（§7.2 加权规则） | **确定性代码** | 规则表翻译成代码，可测可预期；LLM 打分会漂移 |
| 复用阈值 / 去重 / 收敛判定 / 3 次无结果 | **确定性代码** | 行为约束是规则，不是模型判断 |
| claims 提炼 | LLM（extract 链） | 「从文本里挖观点」是软判断 |
| 片段审核 / 矛盾识别 / 修订任务生成 | LLM（critic 链） | 四维审核是语义判断 |
| **LLM 裁决的兜底修正** | **确定性代码覆盖** | LLM 说 PASS 但评分 < 7 → 强制 NEEDS_REVISION |

**一句话**：LLM 做「软判断」，代码做「硬规则」，硬规则是最后防线。

## 3. LLM 调用抽象（可测性设计，同 M6 的 summarize 注入）

智能体不直接拿 factory，而是收一个函数：

```python
llm: Callable[[str, list[ChatMessage]], str]
# 第一个参数 = task_kind（"extract" / "critic"），第二个参数 = 消息列表
# 返回 = 模型输出文本（content）
```

默认值内部包 `factory.complete`（取 `.content`）；**测试传假函数**（按 task_kind 返回固定 JSON），完全确定可控。这就是 M6 `summarize` 注入的同一招。

⚠️ 为什么必须注入：MockProvider 的关键词表只认「拆解/子查询」和「提取/claim」，Critic 的审核 prompt 必然命中 fallback（非 JSON）→ `parse_json` 抛 OutputParseError。所以 **critic 不能依赖 MockProvider**，测试一律注入假 llm。

## 4. 接口契约

### 4.0 先改两处基础设施（用户写）

**settings.py 加两行**：
```python
route_critic: str = "deepseek,mock"              # 审核必须强模型（§5.3）
reuse_similarity_threshold: float = 0.75         # §6.4 记忆复用阈值
```

**core/schemas.py 两处**：
1. `ResearchResult` 加一个字段：`snippets_rejected: int = 0`（写入门槛拒收的计数，默认 0 向后兼容）
2. 加 `RunResult`（orchestrator 输出，§17.4 的 MVP 简化版）：
```python
class RunResult(BaseModel):
    """多任务编排的最终汇总。"""
    topic: str
    tasks: list[Task]                      # 终态（含 reflection_count）
    approved_snippets: list[ResearchSnippet]
    reused_hits: list[ReusedHit]           # 记忆复用命中（复用率统计源）
    contradictions: list[str]              # 未解决矛盾描述（§7.4 存疑标注）
    reflection_rounds: int                 # 实际反思轮次
    errors: list[str]
```

### 4.1 researcher.py —— 研究员（§5.2）

```python
class Researcher:
    """执行检索任务：复用优先 → 搜索 → 提取 → 提炼 → 写入长期记忆。"""
    def __init__(self, settings, tools: ToolBox, memory: MemoryManager,
                 factory: ModelFactory | None = None,
                 llm: Callable[[str, list[ChatMessage]], str] | None = None) -> None:
        # llm 缺省：包 factory.complete；两者都没有 → ConfigError（缺一不可）

    def research(self, task: Task) -> ResearchResult:
        """执行一个任务。流程见 §5.1；连续 3 次无结果 → errors 上报（§5.2 行为约束）。"""
```

**research 流程**：
1. **复用优先**：`memory.recall(task.description, top_k=3)`；`similarity >= reuse_similarity_threshold`（0.75）的命中直接记入 `snippets_reused`，**不重复抓取**（架构原则 2，M9 复用率 ≥60% 的实现根基）
2. **搜索**：对 `task.search_queries`（最多前 2 个）调 `tools.search`；**连续 3 次搜索无结果 → errors.append + 提前终止**（§5.2 行为约束）
3. **提取**：每个结果（最多前 3 个）调 `tools.extract(url)`；ToolError → 记 errors，继续下一个
4. **提炼**：`parse_json(llm("extract", [prompt]), SnippetDraft)` → claims + evidence_excerpt；OutputParseError → 记 errors 跳过
5. **组装 + 写入**：`ResearchSnippet`（元数据来自 SearchResult：url/title/type→publication_status 推断；**可信度初评 = 来源类型分（SOURCE_SCORES 直接取值），不 import critic**——初评粗糙没关系，终评归 critic 的 60/30/10 加权）→ `memory.remember(snippet)`；AddResult 计入 written / duplicate / **rejected（记入 snippets_rejected，news/blog 初评 < 7 会被门槛拒收，这是 §6.4 写入门槛的正确行为）**
6. 返回 `ResearchResult(task, snippets_written, snippets_reused, duplicates_skipped, snippets_rejected, errors)`

**要点**：`_extract_draft(text) -> SnippetDraft | None` 拆成独立方法（可单独测）；prompt 含「提取」关键词（与 MockProvider 关键词表兼容，离线全链路可用）。

### 4.2 critic.py —— 批判者（§5.3/§7）

```python
def score_credibility(source_type: SourceType, published_at: str | None,
                      corroborations: int = 1) -> float:
    """§7.2 加权评分（确定性规则，不调 LLM）。"""

class Critic:
    """结构化反思：LLM 审核 + 规则评分兜底 + 修订任务单。"""
    def __init__(self, settings, memory: MemoryManager,
                 llm: Callable[[str, list[ChatMessage]], str] | None = None) -> None:

    def critique(self, task: Task, snippets: list[ResearchSnippet]) -> CritiqueOutput:
        """审核任务产出。verdict 由 LLM 给出，但被确定性规则修正（评分防线）。"""
```

**score_credibility（§7.2 规则翻译，60/30/10）**：
```python
SOURCE_SCORES = {PAPER: 10, OFFICIAL: 8, PREPRINT: 7, NEWS: 6, BLOG: 4}

source = SOURCE_SCORES[source_type]                  # 权重 60%
cross  = {1: 10, 2: 20, 3: 30}[min(corroborations, 3)]  # 权重 30%（≥3 封顶）
fresh  = 10 if 近2年 else 7 if 2-3年 else 逐年衰减到 0   # 权重 10%；未知 → 5
return round(0.6 * source + 0.3 * cross + 0.1 * fresh, 2)
```

**critique 流程**：
1. **交叉验证统计**（确定性）：同任务内 claims 有重合、且 source_url 不同的片段计数 → 每个片段获得 corroborations
2. **LLM 审核**：一次调用，prompt 包全部片段 + 历史记忆上下文（`memory.recall(task.description, top_k=3)` 做历史兼容性）；要求输出完整 `CritiqueOutput`（verdict / snippet_evaluations / revision_tasks / conflict_details）；`parse_json(content, CritiqueOutput)`
3. **评分防线**（确定性覆盖）：对每个片段算 `score_credibility`（**覆盖** LLM 的 credibility_score）；若任一片段评分 < 7 → **强制** verdict = NEEDS_REVISION（LLM 说 PASS 也改），并确保 revision_tasks 非空
4. 返回修正后的 CritiqueOutput

### 4.3 orchestrator.py —— 协调器（§5.5/§7.4）

```python
class Orchestrator:
    """任务编排：Researcher ↔ Critic 循环 + 收敛降级。纯 Python 循环，不依赖 LangGraph。"""
    def __init__(self, settings, tools: ToolBox, memory: MemoryManager,
                 factory: ModelFactory | None = None, llm: Callable | None = None) -> None:

    def run(self, tasks: list[Task]) -> RunResult:
        """执行任务列表（按优先级 + 依赖顺序），汇总终态。"""
```

**run 流程（§7.4 收敛规则）**：
```
队列 = tasks 按 (priority, 依赖已完成的优先) 排序
while 队列非空:
    task = 出队
    task.status = RUNNING
    result = researcher.research(task)                  # 检索
    critique = critic.critique(task, result.snippets_written)

    if verdict == PASS:
        task.status = COMPLETED
        approved += result.snippets_written
    elif task.reflection_count < 2:                     # 最多 2 轮反思（§7.4）
        task.reflection_count += 1
        revision_tasks 转成新 Task（source=CRITIC_REVISION, 挂依赖）入队
        task 重新入队（等修订任务完成后再审一次）
    else:                                               # 2 轮仍不过 → 降级
        task.status = COMPLETED
        approved += result.snippets_written             # 「存疑」放行，contradictions 记录
    record_event("agent_step", ...)                     # 全链路可观测

reused_hits = 汇总所有 research 的 snippets_reused
return RunResult(topic, tasks, approved, reused_hits, contradictions, reflection_rounds, errors)
```

**要点**：
- 依赖检查：任务出队时若 dependencies 中有未 COMPLETED 的 → 暂回队列（MVP 简化：无环假设，不做拓扑排序）
- `llm` 参数透传给 Researcher/Critic（一个假函数喂两个 agent）
- reflection_count 来自 Task 自带字段（§17.1，M2 就写好了，直接用！）

## 5. 设计决策表

| # | 决策 | 原因 |
|---|---|---|
| ① | 评分规则/阈值/收敛判定是确定性代码 | 规则表可测；LLM 打分漂移；「硬规则是最后防线」 |
| ② | 智能体收 `llm: Callable` 而不是 factory | 测试注入假函数（同 M6 summarize）；critic 不能依赖 MockProvider 关键词表 |
| ③ | Critic 一次 LLM 调用出完整 CritiqueOutput | 一个 schema 一个调用，parse 一次；评分防线负责修正 |
| ④ | 复用优先（0.75 阈值）| 架构原则 2；M9 复用率 ≥60% 的实现根基；省 API 调用 |
| ⑤ | 连续 3 次无结果 → errors 上报 | §5.2 行为约束（防白耗 LLM） |
| ⑥ | 2 轮反思后仍不过 → 降级「存疑放行」 | §7.4：避免死循环；报告阶段标注存疑 |
| ⑦ | orchestrator 纯 Python 循环，不用 LangGraph | MVP 可测性优先；状态机不是必须框架 |
| ⑧ | mock 全链路只保证 researcher（extract 链）| MockProvider 只认「提取」关键词；critic 测试注入假 llm |

## 6. 验收标准（我的测试会验证这些）

**researcher（假 llm + mock 工具 + mock embedder，临时目录）**：
- [ ] 正常流程：搜索 → 提取 → 提炼 → 写入记忆，ResearchResult 计数正确
- [ ] 复用优先：记忆里已有高相似片段 → 记入 reused，不再调用工具
- [ ] 连续 3 次搜索无结果 → errors 含「连续 3 次」，提前终止
- [ ] 提取失败 / 提炼解析失败 → 记 errors，继续其他结果，不中断
- [ ] duplicate 片段计入 duplicates_skipped，不写入
- [ ] 低可信度片段 → rejected 计数

**critic**：
- [ ] score_credibility 规则表：paper 满时效 → 10.0；blog + 未知时间 + 单一来源 → 低分；边界值精确
- [ ] LLM 说 PASS 但某片段评分 < 7 → 强制 NEEDS_REVISION + revision_tasks 非空
- [ ] 全部高分 → PASS 原样通过
- [ ] 交叉验证统计：同 claims 不同来源的片段 corroborations ≥ 2

**orchestrator（全 mock + 假 llm）**：
- [ ] 一轮 PASS → 全部 approved，无 revision 入队
- [ ] NEEDS_REVISION → 修订任务入队（source=CRITIC_REVISION），二轮通过后 approved
- [ ] 2 轮仍不过 → 降级放行（approved 含存疑片段 + contradictions 记录），不死循环
- [ ] 依赖顺序：被依赖任务先完成
- [ ] RunResult 汇总：reflection_rounds / reused_hits / errors 正确
- [ ] 全链路 mock 离线可跑（researcher 走真实 MockProvider 也行）

## 7. 写作顺序

1. **基础设施**：settings 两行 + schemas 的 RunResult（5 分钟）
2. **researcher.py**（最核心：复用 → 检索 → 提炼 → 写入）
3. **critic.py**（评分规则 + 审核 + 修正）
4. **orchestrator.py**（把前两个串成循环）