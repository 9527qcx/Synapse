"""M7 智能体测试共享设施：可编程假 llm + 假工具 + mock 设置。

为什么需要：
- FakeLLM：critic 不能依赖 MockProvider（关键词表不认审核 prompt）→ 按脚本返回 verdict
- FakeTools：mock 工具搜索结果固定，跨轮会被去重 → 假工具每次返回新材料并记录调用次数
"""
from __future__ import annotations

import json

import pytest

from synapselib.config.settings import Settings
from synapselib.core.errors import ToolError
from synapselib.tools.schemas import ExtractionResult, SearchResponse, SearchResult


class FakeLLM:
    """可编程假 llm：extract 固定返回草稿；critic 按 verdicts 脚本返回。

    - verdicts: critic 调用序列（超出用最后一个）
    - revision_desc: NEEDS_REVISION 时生成的修订任务描述；prompt 含它的任务
      （即修订任务）默认 PASS，避免 orchestrator 测试里修订链无限延伸
    - evals: 附加到 critic 输出的 snippet_evaluations（测防线覆盖/补齐）
    - bad_extract: extract 返回非 JSON（测提炼解析失败容错）
    """

    def __init__(self, verdicts=None, revision_desc="补充权威来源",
                 pass_on_revision=True, evals=None, bad_extract=False):
        self.verdicts = list(verdicts or ["PASS"])
        self.revision_desc = revision_desc
        self.pass_on_revision = pass_on_revision
        self.evals = list(evals or [])
        self.bad_extract = bad_extract
        self.critic_calls = 0
        self._idx = 0

    def __call__(self, task_kind, messages):
        if task_kind == "extract":
            if self.bad_extract:
                return "这不是 JSON"
            return json.dumps({
                "claims": ["RAG 引入外部知识减少幻觉", "幻觉率评估依赖基准数据集"],
                "evidence_excerpt": "（mock）检索增强可显著降低幻觉发生率。",
            }, ensure_ascii=False)
        self.critic_calls += 1
        prompt = messages[0].content
        if self.pass_on_revision and self.revision_desc in prompt:
            verdict, revisions = "PASS", []
        else:
            verdict = self.verdicts[min(self._idx, len(self.verdicts) - 1)]
            self._idx += 1
            if verdict == "NEEDS_REVISION":
                revisions = [{"task_description": self.revision_desc,
                              "search_queries": ["补充检索"], "target_info": [],
                              "priority": 7}]
            else:
                revisions = []
        return json.dumps({
            "verdict": verdict, "overall_feedback": "ok",
            "snippet_evaluations": self.evals,
            "revision_tasks": revisions, "conflict_details": [],
        }, ensure_ascii=False)


class FakeTools:
    """假工具：search 每次返回新材料（round 递增的 arxiv URL），记录调用次数。

    - empty=True: 永远返回空结果（测「连续 3 次无结果」）
    - fail_extract=True: extract 抛 ToolError（测错误不中断）
    """

    def __init__(self, empty=False, fail_extract=False):
        self.round = 0
        self.empty = empty
        self.fail_extract = fail_extract
        self.search_calls = 0
        self.extract_calls = 0

    def search(self, query):
        self.search_calls += 1
        if self.empty:
            return SearchResponse(query=query, results=[])
        self.round += 1
        return SearchResponse(query=query, results=[SearchResult(
            title=f"材料 R{self.round}",
            url=f"https://arxiv.org/abs/2312.1099{self.round % 10}",
            snippet="mock 摘要", published_at="2025-01-01")])

    def extract(self, url):
        self.extract_calls += 1
        if self.fail_extract:
            raise ToolError(f"提取失败 {url}")
        return ExtractionResult(url=url, title="t",
                                text="RAG 引入外部知识减少幻觉的正文", truncated=False)


@pytest.fixture
def mock_settings(tmp_path):
    """mock 全开：工具走假数据（ToolBox 需要 mock_mode）+ 临时 Chroma 目录。"""
    return Settings(mock_mode=True, embedder="mock",
                    chroma_dir=str(tmp_path / "chroma"),
                    preferences_file=str(tmp_path / "prefs.json"))


@pytest.fixture
def make_llm():
    return FakeLLM


@pytest.fixture
def make_tools():
    return FakeTools
