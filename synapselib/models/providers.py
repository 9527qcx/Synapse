"""模型提供方：DeepSeek（云端）/ Ollama（本地）/ Mock（测试）。

统一契约 ChatProvider：所有 provider 的 complete() 签名完全一致，
ModelFactory 才能做「降级链」——一个挂了换下一个，调用方无感知。
"""
from __future__ import annotations

import json
import time
from typing import Protocol

import httpx

from ..core.errors import ProviderError
from .schemas import ChatMessage, ChatRole, ModelResponse, Usage


class ChatProvider(Protocol):
    """模型提供方协议：三个 provider 都要满足这个形状。"""

    name: str

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> ModelResponse: ...


class DeepSeekProvider:
    """DeepSeek 云端 API（OpenAI 兼容协议）。

    构造参数由 ModelFactory 从 settings 传入，provider 自身不碰配置——
    这叫「依赖注入」：不在内部 import 全局配置，而是外部把参数递进来。
    好处：测试时可以用假参数构造，不依赖 .env，也容易替换。
    """

    name = "deepseek"

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        # 延迟导入：只有真的要用 deepseek 时才加载 openai SDK
        from openai import OpenAI

        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> ModelResponse:
        # ChatMessage.role 是 str 枚举，显式取 .value 转成纯字符串，
        # 避免把枚举对象直接交给 SDK 序列化（能跑，但依赖隐式行为，不清晰）
        payload = [{"role": m.role.value, "content": m.content} for m in messages]

        kwargs: dict = dict(
            model=self._model,
            messages=payload,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            # DeepSeek 的 json_mode 约束：prompt 里必须出现 "json" 字样，
            # 否则报 400。这个约束由调用方（prompt 模板）保证，这里只传参。
            kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            # 把 openai SDK 的各种异常（连接/限流/鉴权/状态码）
            # 统一映射成 ProviderError —— 降级链就靠它「接住」失败。
            # openai.APIError 是所有 SDK 异常的基类，一次捕获全覆盖。
            import openai

            if isinstance(e, openai.APIError):
                raise ProviderError(f"DeepSeek 调用失败: {e}") from e
            raise  # 非 SDK 异常（如编码错误）是 bug，让它直接炸出来

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        content = resp.choices[0].message.content or ""
        usage = resp.usage
        return ModelResponse(
            content=content,
            model=resp.model or self._model,
            provider=self.name,
            usage=Usage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )
            if usage
            else None,
            latency_ms=latency_ms,
        )


class OllamaProvider:
    """Ollama 本地模型（HTTP API，非 OpenAI 协议，用 httpx 直连）。

    API 端点：POST {base_url}/api/chat
    响应结构：{"message": {"role", "content"}, "prompt_eval_count", "eval_count", ...}
    """

    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 30.0) -> None:
        self._base_url = base_url
        self._model = model
        # Client 构造一次复用（你的原稿每次调用新建，浪费连接），
        # timeout 必须有——否则网络不通时请求无限挂起
        self._client = httpx.Client(timeout=timeout)

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> ModelResponse:
        # 与 DeepSeek 同样的道理：ChatMessage 转成纯 dict 再交给 httpx
        payload = [{"role": m.role.value, "content": m.content} for m in messages]

        body: dict = {
            "model": self._model,
            "messages": payload,
            "stream": False,
            # Ollama 的参数在 options 里嵌套；num_predict 对应 max_tokens
            "options": {"temperature": temperature, "num_predict": max_tokens},
            # json_mode → format: "json"（Ollama 的强制 JSON 输出）
            "format": "json" if json_mode else None,
        }

        start = time.perf_counter()
        try:
            resp = self._client.post(f"{self._base_url}/api/chat", json=body)
            resp.raise_for_status()
        except httpx.RequestError as e:
            # httpx.RequestError 是连接类异常的基类（超时/拒绝连接/DNS）
            raise ProviderError(f"Ollama 调用失败: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"Ollama 返回非 2xx: {e.response.status_code} {e.response.text[:200]}"
            ) from e

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        data = resp.json()
        message = data.get("message", {})
        return ModelResponse(
            content=message.get("content", ""),
            model=data.get("model") or self._model,
            provider=self.name,
            usage=Usage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            ),
            latency_ms=latency_ms,
        )


class MockProvider:
    """确定性假响应 —— 不联网、不看模型，按关键词「查表」返回固定答案。

    与真 Provider 的本质区别：
      真 Provider = 发请求 → 拿真实响应
      Mock       = 看输入 → 查表返回固定 JSON（永远同一份，测试可预期）
    用途：单元测试 / 离线调试 / 降级链的最终兜底。
    """

    name = "mock"

    # ---- 模块级常量：固定响应（ensure_ascii=False 让中文可读）----
    _QUERY_RESPONSE = json.dumps(
        {
            "queries": [
                "大模型幻觉缓解技术的核心方法",
                "RLHF 与幻觉的关系",
                "RAG 与幻觉的关系",
                "幻觉评估基准",
            ]
        },
        ensure_ascii=False,
    )
    _EXTRACT_RESPONSE = json.dumps(
        {
            "claims": [
                "RLHF 通过人类反馈优化降低幻觉",
                "RAG 引入外部知识减少幻觉",
                "幻觉率评估依赖基准数据集",
            ],
            "evidence_excerpt": "（mock 摘录）示例：研究表明检索增强可显著降低幻觉发生率。",
        },
        ensure_ascii=False,
    )
    _FALLBACK_TEXT = "（mock 响应）固定回复：模型未识别任务类型。"

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> ModelResponse:
        # 只看最后一条 user 消息做路由；其余参数（temperature 等）一律忽略
        last_user = next(
            (m.content for m in reversed(messages) if m.role == ChatRole.USER), ""
        )
        if any(k in last_user for k in ("拆解", "子查询", "query")):
            content = self._QUERY_RESPONSE
        elif any(k in last_user for k in ("提取", "claim")):
            content = self._EXTRACT_RESPONSE
        else:
            content = self._FALLBACK_TEXT

        return ModelResponse(
            content=content,
            model="mock-model",
            provider=self.name,
            usage=None,
            latency_ms=0.0,
        )
