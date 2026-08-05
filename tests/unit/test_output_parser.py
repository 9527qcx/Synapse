"""core/output_parser.py 单元测试：四种解析路径 + 失败降级。"""
from __future__ import annotations

import pytest

from synapselib.core.errors import OutputParseError
from synapselib.core.output_parser import parse_json
from synapselib.core.schemas import SubQueryPlan


class TestParseJson:
    def test_pure_json(self):
        out = parse_json('{"queries": ["q1", "q2"]}', SubQueryPlan)
        assert out.queries == ["q1", "q2"]

    def test_fenced_code_block(self):
        raw = '```json\n{"queries": ["q1"]}\n```'
        assert parse_json(raw, SubQueryPlan).queries == ["q1"]

    def test_fenced_without_lang(self):
        raw = "```\n{\"queries\": [\"q1\"]}\n```"
        assert parse_json(raw, SubQueryPlan).queries == ["q1"]

    def test_prose_before_json(self):
        raw = '好的，以下是拆解结果：\n{"queries": ["q1", "q2", "q3"]}\n希望对你有帮助。'
        out = parse_json(raw, SubQueryPlan)
        assert len(out.queries) == 3

    def test_garbage_raises(self):
        with pytest.raises(OutputParseError):
            parse_json("完全不是 JSON 的内容", SubQueryPlan)

    def test_empty_raises(self):
        with pytest.raises(OutputParseError):
            parse_json("   ", SubQueryPlan)

    def test_valid_json_wrong_schema_raises(self):
        # JSON 合法但缺字段（queries 是必填）
        with pytest.raises(OutputParseError) as exc:
            parse_json('{"wrong_key": 1}', SubQueryPlan)
        assert "queries" in str(exc.value)
