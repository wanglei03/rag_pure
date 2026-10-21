"""Embedding 抽象层:同一接口,支持 OpenAI 兼容端点 + 离线确定性 mock。

面试点:embedding 服务为什么通常要和 LLM 分开配置?——很多供应商只提供其中一类
(例如 DeepSeek 只有 chat 没有 embedding),生产上经常是"chat 用 A 家、embedding 用 B 家"。
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    dim: int = 0

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """返回 shape=(len(texts), dim) 的 L2 归一化向量"""


class OpenAICompatibleEmbedder(BaseEmbedder):
    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=120)
        self._model = model
        r = self._client.embeddings.create(model=model, input=["dimension_probe"])
        self.dim = len(r.data[0].embedding)

    def encode(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        return _l2_normalize(vecs)


class MockEmbedder(BaseEmbedder):
    """离线确定性 embedding:对每个分词后的 token 做哈希随机投影再求和归一化。

    性质:共享 token 越多的两段文本,余弦相似度越高 —— 足够演示检索管线的全部机制。
    向量维度 128,任何机器上结果可复现。
    """

    dim = 128

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i] = self._encode_one(text)
        return _l2_normalize(out)

    def _encode_one(self, text: str) -> np.ndarray:
        import jieba

        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = [t for t in jieba.lcut(text.lower()) if t.strip()]
        for tok in tokens:
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        if not np.any(vec):  # 空文本兜底
            vec[0] = 1.0
        return vec


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def build_embedder() -> BaseEmbedder:
    from config import settings

    cfg = settings.get_embed_config()
    if cfg["provider"] == "mock":
        return MockEmbedder()
    if not cfg.get("api_key"):
        raise RuntimeError(
            f"provider={cfg['provider']} 需要 API Key,请在 .env 中设置对应 *_API_KEY,或改用 EMBED_PROVIDER=mock"
        )
    print(f"[build_embedder] provider={cfg['provider']}, model={cfg['model']}, base_url={cfg['base_url']}")
    return OpenAICompatibleEmbedder(base_url=cfg["base_url"], api_key=cfg["api_key"], model=cfg["model"])
