"""LLM 输出解析器 —— 大纲 §13.4 OutputFixingParser 的轻量落地。

即使要求输出 JSON，LLM 仍可能返回：代码围栏包裹、前导散文、前后多余文本。
本模块按「从宽到严」的策略提取并校验，全部失败抛 OutputParseError（调用方降级处理）。
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from .errors import OutputParseError


def parse_json(content: str, schema: type[BaseModel]) -> BaseModel:
    """从 LLM 输出中解析出符合 schema 的对象。

    参数:
        content: LLM 返回的原始文本
        schema: 目标 Pydantic 模型（如 SubQueryPlan）
    返回:
        校验通过的对象实例
    抛出:
        OutputParseError: 提取失败或校验失败（错误信息含字段明细）
    """
    obj = _extract_json(content)
    try:
        return schema.model_validate(obj)
    except ValidationError as e:
        details = "; ".join(f"{err['loc'][0]}: {err['msg']}" for err in e.errors())
        raise OutputParseError(f"JSON 结构校验失败: {details}") from e


def _extract_json(content: str) -> Any:
    """从原始文本中提取 JSON 对象/数组。依次尝试四种策略。"""
    text = content.strip()
    if not text:
        raise OutputParseError("LLM 输出为空")

    # 策略 1：直接解析（最理想的情况）
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 策略 2：剥离 ```json ... ``` 代码围栏
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return _loads_or_raise(fenced.group(1).strip(), "代码围栏内内容不是合法 JSON")

    # 策略 3：截取首个 { 到最后一个 }（LLM 常在 JSON 前写散文）
    obj_match = re.search(r"\{", text), re.search(r"\}", text[::-1])
    if obj_match[0] and obj_match[1]:
        start, end = obj_match[0].start(), len(text) - obj_match[1].start()
        candidate = text[start:end]
        if end > start:
            return _loads_or_raise(candidate, "提取的 JSON 对象不合法")

    # 策略 4：截取首个 [ 到最后一个 ]（数组场景）
    arr_match = re.search(r"\[", text), re.search(r"\]", text[::-1])
    if arr_match[0] and arr_match[1]:
        start, end = arr_match[0].start(), len(text) - arr_match[1].start()
        candidate = text[start:end]
        if end > start:
            return _loads_or_raise(candidate, "提取的 JSON 数组不合法")

    preview = text[:200].replace("\n", " ")
    raise OutputParseError(f"无法从 LLM 输出中提取 JSON，内容开头: {preview!r}")


def _loads_or_raise(raw: str, message: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise OutputParseError(f"{message}（位置 {e.pos}）") from e
