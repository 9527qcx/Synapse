"""环境自检脚本（Windows 排障利器）。

用法：
    python scripts/check_env.py

输出 [OK]/[MISSING]/[WARN]，逐项排查环境问题：
- [MISSING] 说明缺依赖或配置加载失败，需要修复
- [WARN]    不影响核心链路（如 Ollama 未运行）
"""
from __future__ import annotations

import sys
from pathlib import Path

# Windows 控制台中文乱码修复：cp936(GBK) 控制台强制切 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 保证未安装为包时也能 import synapselib
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synapselib.config.settings import get_settings  # noqa: E402


def _check(name: str, fn) -> None:
    try:
        fn()
        print(f"  [OK]      {name}")
    except Exception as e:  # noqa: BLE001
        print(f"  [MISSING] {name}: {e}")


def main() -> None:
    print(f"Python     : {sys.version.split()[0]}  ({sys.executable})")
    print(f"项目根目录  : {ROOT}")
    print()

    # 1. 关键依赖逐个导入验证
    deps = ("pydantic", "pydantic_settings", "chromadb", "openai",
            "arxiv", "trafilatura", "langfuse", "tiktoken")
    for mod in deps:
        _check(f"依赖 {mod}", lambda m=mod: __import__(m))

    # 2. 配置加载
    try:
        settings = get_settings()
        key_filled = bool(settings.deepseek_api_key and "在此粘贴" not in settings.deepseek_api_key)
        print(f"  [OK]      配置加载（DeepSeek key 已填: {key_filled}）")
        print(f"            mock_mode: {settings.mock_mode}, "
              f"嵌入: {settings.embedder}, 路由: {settings.routes}")
    except Exception as e:  # noqa: BLE001
        print(f"  [MISSING] 配置加载: {e}")
        return

    # 3. 嵌入模型（heavy 库延迟导入，避免自检卡住）
    if settings.embedder == "local":
        _check("嵌入依赖 sentence-transformers", lambda: __import__("sentence_transformers"))
    else:
        print("  [INFO]    嵌入模式 = mock（无需下载模型权重）")

    # 4. Ollama 状态
    import urllib.request

    try:
        urllib.request.urlopen(f"{settings.ollama_base_url}/api/tags", timeout=3)
        print("  [OK]      Ollama 服务运行中")
    except Exception:  # noqa: BLE001
        print("  [WARN]    Ollama 未运行（本地模型不可用，路由会自动跳过它）")

    print()
    print("自检结束：没有 [MISSING] 即为健康环境；[WARN] 不影响核心链路。")


if __name__ == "__main__":
    main()
