"""全局配置：从环境变量 / .env 文件加载（pydantic-settings）。

设计要点：
- 所有配置项统一前缀 SYNAPSE_，例如 SYNAPSE_MOCK_MODE=1
- .env 文件位于项目根目录，已被 .gitignore 排除，密钥不会入库
- 相对路径（如 chroma_dir）一律基于项目根目录解析，避免启动目录漂移
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：本文件位于 synapselib/config/，向上两级即项目根
ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """全部配置项。环境变量统一前缀 SYNAPSE_，例如 SYNAPSE_MOCK_MODE=1。"""

    model_config = SettingsConfigDict(
        env_prefix="SYNAPSE_",
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 大模型 ----
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # ---- 模型路由：按任务难度配置降级链（逗号分隔，从左到右尝试）----
    route_plan: str = "deepseek,ollama,mock"  # 规划类任务（§5.1 建议本地模型，可切）
    route_extract: str = "deepseek,ollama,mock"  # 检索提取类
    route_summarizer: str = "deepseek,mock"  # 短期记忆摘要

    # ---- 嵌入模型 ----
    embedder: str = "local"  # local = BGE-M3；mock = 测试嵌入（无需下载）
    embedder_model: str = "BAAI/bge-m3"

    # ---- 向量记忆（ChromaDB）----
    chroma_collection_name: str = "snippets"
    chroma_dir: str = "data/chroma"
    dedup_cosine_threshold: float = 0.90  # §6.2 去重粗筛阈值
    dedup_claim_overlap: float = 0.80  # §6.2 去重细判阈值（claims 重合率）
    credibility_score_threshold: float = 7.0  # §6.2 信度阈值（0-10）

    # ---- 短期记忆 ----
    stm_budget_tokens: int = 8000  # 超过该 token 数触发摘要压缩
    stm_keep_last_turns: int = 6  # 保留最近 N 轮完整对话

    # ---- 用户偏好（§6.5 MVP：显式反馈，JSON 存储）----
    preferences_file: str = "data/user_preferences.json"

    # ---- 可观测（LangFuse）----
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ---- 工具 ----
    search_base_url: str = "http://localhost:8080"  # SearXNG

    # ---- 调试 ----
    mock_mode: bool = False  # True = 全链路 mock，离线可跑通

    # ---------- 派生属性 ----------

    @property
    def routes(self) -> dict[str, list[str]]:
        """任务难度 → provider 降级链（从左到右尝试，全部失败则抛 ModelError）。"""
        return {
            "plan": self._parse_chain(self.route_plan),
            "extract": self._parse_chain(self.route_extract),
            "summarizer": self._parse_chain(self.route_summarizer),
        }

    @staticmethod
    def _parse_chain(raw: str) -> list[str]:
        return [p.strip() for p in raw.split(",") if p.strip()]

    @property
    def langfuse_enabled(self) -> bool:
        """public + secret 两个 key 齐全才算启用（守卫式，缺一即 no-op）。"""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    def chroma_path(self) -> Path:
        """ChromaDB 持久化目录。相对路径基于项目根解析，保证从任何目录启动行为一致。"""
        p = Path(self.chroma_dir)
        return p if p.is_absolute() else ROOT_DIR / p

    def preferences_path(self) -> Path:
        """用户偏好 JSON 路径（同 chroma_path 的相对路径解析规则）。"""
        p = Path(self.preferences_file)
        return p if p.is_absolute() else ROOT_DIR / p


_settings: Settings | None = None

def get_settings() -> Settings:
    """单例获取配置（避免重复读 .env / 环境变量）。"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
