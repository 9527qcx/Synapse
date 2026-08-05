"""项目统一异常层级。

所有自定义异常都继承 SynapseError，调用方只需捕获基类即可覆盖全部业务异常；
系统异常（KeyError 等）不在本层级，捕获时按需显式写出。
"""
from __future__ import annotations


class SynapseError(Exception):
    """项目所有自定义异常的基类。"""


class ConfigError(SynapseError):
    """配置错误：环境变量缺失、非法值等。"""


class ModelError(SynapseError):
    """模型调用失败：降级链全部耗尽。"""


class ProviderError(ModelError):
    """单个模型提供方调用失败（DeepSeek/Ollama/Mock 之一）。"""


class OutputParseError(SynapseError):
    """LLM 输出解析失败：JSON 提取或 Schema 校验未通过。"""


class ToolError(SynapseError):
    """工具调用失败（搜索、论文检索、网页提取等）。"""


class RetryExhaustedError(ToolError):
    """重试耗尽：达到最大重试次数仍失败。"""


class MemoryError(SynapseError):
    """记忆层错误（向量库读写、去重等）。"""
