"""模型工厂 —— 模型层的枢纽：路由 + 降级链。

设计哲学（大纲 §3「模型路由：简单任务用本地模型，Critic/Synthesis 用强模型」落地）：

1. **路由**：按任务难度查 settings.routes，决定「这次调用该用哪些 provider」
2. **降级**：链路从左到右尝试，ProviderError 是「此路不通」的**信号**，
   工厂接住信号后换下一个 —— 调用方（智能体）完全无感知，
   不知道也不关心背后换了几次模型

整个文件就一个核心模式，记住它：
    for name in chain:
        try: 尝试 → 成功就 return
        except 信号: 记原因 → 试下一个
    全失败 → ModelError(带全部原因)
"""
import logging

from ..config.settings import Settings
from ..core.errors import ConfigError, ModelError, ProviderError
from ..observability.tracing import record_event
from .providers import ChatProvider, DeepSeekProvider, OllamaProvider, MockProvider
from .schemas import ChatMessage, ModelResponse

logger = logging.getLogger(__name__)


class ModelFactory:
    """模型工厂：路由 + 降级链。

    - 构造时只收 Settings（配置来源），不碰任何网络
    - provider 实例懒加载缓存：同一个 provider 只构造一次，复用连接
      （OpenAI 客户端/httpx Client 内部都有连接池，反复构造是浪费）
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._providers: dict[str, ChatProvider] = {}  # 懒加载缓存：name → 实例

    # ------------------------------------------------------------------ 实例获取

    def get_provider(self, name: str) -> ChatProvider:
        """按名字获取 provider 实例（懒加载 + 缓存）。

        关键设计：**不存在的依赖要抛 ProviderError，而不是构造一个必挂的客户端**。
        例：DeepSeek key 为空 → 这里直接抛 ProviderError，
        complete() 的降级循环接住这个信号后就会自动滑向下一个 provider。
        """
        # 缓存命中：直接复用（连接池、模型句柄都不重建）
        if name in self._providers:
            return self._providers[name]

        # 缓存未命中：按名字构造（依赖注入 —— 参数全部来自 settings，provider 不碰配置）
        if name == DeepSeekProvider.name:
            api_key = self._settings.deepseek_api_key
            if not api_key:
                # ⚠️ 信号设计：key 没配 = 这条路走不通，立即发信号让降级链跳过
                raise ProviderError("DeepSeek API key 未配置（去 .env 填写）")
            provider = DeepSeekProvider(
                api_key=api_key,
                base_url=self._settings.deepseek_base_url,
                model=self._settings.deepseek_model,
            )
        elif name == OllamaProvider.name:
            provider = OllamaProvider(
                base_url=self._settings.ollama_base_url,
                model=self._settings.ollama_model,
            )
        elif name == MockProvider.name:
            provider = MockProvider()  # Mock 无外部依赖，不需要任何参数
        else:
            # 配置/代码写错了 provider 名 → 这是编程错误，用 ConfigError 明确表达
            raise ConfigError(f"未知 provider: {name}")

        # 构造成功才入缓存（失败的不会缓存，下次重试仍走完整流程）
        self._providers[name] = provider
        return provider

    # ------------------------------------------------------------------ 统一入口

    def complete(
        self,
        task_kind: str,
        messages: list[ChatMessage],
        **kwargs,
    ) -> ModelResponse:
        """按任务类型路由调用模型，内部处理降级。

        参数:
            task_kind: 任务难度标识（plan / extract / summarizer），
                       决定走哪条降级链（查 settings.routes）
            messages:  对话消息列表
            **kwargs:  透传给 provider 的 complete（temperature、json_mode 等）

        返回:
            第一个成功 provider 的响应（调用方无感知内部降级）

        抛出:
            ConfigError: task_kind 不在路由表里（且非 mock_mode）
            ModelError:  整条降级链全部失败（消息里带每个环节的失败原因）
        """
        # 1. 选链路
        #    mock_mode = 调试短路：不管路由表，直接 mock（离线、零成本、确定性）
        #    正常模式：查路由表；查不到 → 配置错误
        chain = ["mock"] if self._settings.mock_mode else self._settings.routes.get(task_kind, [])
        if not chain:
            raise ConfigError(f"Unknown task kind: {task_kind}")
        
        # 2. 降级循环（本文件的核心模式）
        failures: list[str] = []  # 收集每一环的失败原因，最终拼进 ModelError
        for i, name in enumerate(chain):
            try:
                provider = self.get_provider(name)
                return provider.complete(messages, **kwargs)  # 成功：直接返回，绝不回头
            except ProviderError as e:
                # 收到「此路不通」信号：记录原因，试下一个
                failures.append(f"{name}: {e}")
                logger.warning("Provider [%s] 失败，降级到下一个: %s", name, e)
                record_event(  # 可观测：每次降级在 LangFuse 面板留痕
                    "provider_fallback",
                    task_kind=task_kind,
                    from_provider=name,
                    to_provider=chain[i + 1] if i + 1 < len(chain) else "无",
                )

        # 3. 整条链都失败：把每一环的原因拼起来抛出去（调试就靠这条消息）
        raise ModelError(f"降级链耗尽 [{task_kind}]: " + " | ".join(failures))
