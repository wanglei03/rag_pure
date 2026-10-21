"""集中配置:Provider 可切换 + 离线 mock,保证项目在任何机器上都能端到端跑通。

切换模型只改 .env 里的 LLM_PROVIDER / EMBED_PROVIDER,业务代码零改动 ——
这也是面试可以讲的点:多供应商适配层怎么设计。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# ---- 路径 ----
DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", BASE_DIR / "data" / "docs"))
STORE_PATH = Path(os.getenv("RAG_STORE_PATH", BASE_DIR / "data" / "vector_store.npz"))
EVAL_SET_PATH = Path(os.getenv("RAG_EVAL_SET", BASE_DIR / "data" / "eval" / "questions.json"))

# ---- 切分与检索参数 ----
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "280"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "60"))
RETRIEVE_TOP_K = int(os.getenv("RAG_RETRIEVE_TOP_K", "5"))
HYBRID_CANDIDATE_K = int(os.getenv("RAG_HYBRID_CANDIDATE_K", "20"))  # 混合检索每路先取的候选数

# ---- Provider 预设:全部走 OpenAI 兼容接口,一套客户端代码切换 ----
PROVIDERS: dict[str, dict] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "llm_model": "deepseek-chat",
        # 注意:DeepSeek 官方没有 embedding 接口,需要 EMBED_PROVIDER 另配 dashscope/ollama/mock
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "rerank_url": "https://maas.qianwenaiapi.com/api/v1/services/rerank/text-rerank/text-rerank",
        "api_key_env": "DASHSCOPE_API_KEY",
        "llm_model": "qwen-plus",
        "embed_model": "text-embedding-v3",
        "rerank_model": "gte-rerank-v2",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "",  # ollama 本地不校验 key
        "llm_model": "qwen2.5:7b",
        "embed_model": "nomic-embed-text",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key_env": "SILICONFLOW_API_KEY",
        "llm_model": "deepseek-ai/DeepSeek-V3",
        "embed_model": "BAAI/bge-m3",
        "rerank_model": "BAAI/bge-reranker-v2-m3",
    },
}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "mock")
RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "overlap")  # overlap(本地) | dashscope | siliconflow


def _preset(name: str) -> dict:
    if name not in PROVIDERS:
        raise ValueError(f"未知 provider: {name},可选: {list(PROVIDERS)} 或 mock")
    return PROVIDERS[name]


def get_llm_config() -> dict:
    if LLM_PROVIDER == "mock":
        return {"provider": "mock", "model": "mock-llm"}
    p = _preset(LLM_PROVIDER)
    env_name = p.get("api_key_env", "")
    api_key = os.getenv(env_name, "") if env_name else "ollama"
    return {
        "provider": LLM_PROVIDER,
        "base_url": p["base_url"],
        "api_key": api_key,
        "model": os.getenv("RAG_LLM_MODEL", p["llm_model"]),
    }


def get_embed_config() -> dict:
    if EMBED_PROVIDER == "mock":
        return {"provider": "mock", "model": "mock-embed", "dim": 128}
    p = _preset(EMBED_PROVIDER)
    if "embed_model" not in p:
        raise ValueError(f"provider {EMBED_PROVIDER} 没有预设 embedding 模型,请设置 RAG_EMBED_MODEL")
    env_name = p.get("api_key_env", "")
    api_key = os.getenv(env_name, "") if env_name else "ollama"
    return {
        "provider": EMBED_PROVIDER,
        "base_url": p["base_url"],
        "api_key": api_key,
        "model": os.getenv("RAG_EMBED_MODEL", p["embed_model"]),
    }


def get_rerank_config() -> dict:
    """overlap 为本地零依赖实现;dashscope/siliconflow 走 HTTP rerank 接口。"""
    if RERANK_PROVIDER == "overlap":
        return {"provider": "overlap"}
    p = _preset(RERANK_PROVIDER)
    if "rerank_model" not in p:
        raise ValueError(f"provider {RERANK_PROVIDER} 没有预设 rerank 模型")
    env_name = p.get("api_key_env", "")
    return {
        "provider": RERANK_PROVIDER,
        "base_url": p.get("rerank_url", p["base_url"]),
        "api_key": os.getenv(env_name, ""),
        "model": os.getenv("RAG_RERANK_MODEL", p["rerank_model"]),
    }


def ensure_dirs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
