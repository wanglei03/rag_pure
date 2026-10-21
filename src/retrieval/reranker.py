"""重排序:召回(top_k 较大)之后再精排,是 RAG 性价比最高的优化之一。

默认 OverlapReranker 为本地零依赖实现(jieba 词 + 字符 bigram 的加权重合度),
配真实模型时切换 DashScopeReranker / SiliconFlowReranker —— 工厂函数按配置选择。

面试点:为什么不在向量库上直接排准?——双塔/向量模型是"_QUERY 和 DOC 分开编码",
为了速度牺牲了交互;reranker 是交叉编码器(query 和 doc 一起进模型),更准但更慢,
所以放在召回之后只对几十个候选精排。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import jieba

from src.vectorstore.store import ScoredChunk


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...


class OverlapReranker(BaseReranker):
    """本地确定性实现:query 词项(含字符 bigram)在 chunk 中的覆盖率,加权词长。"""

    def _terms(self, text: str) -> set[str]:
        words = {t for t in jieba.lcut(text.lower()) if len(t.strip()) > 1}
        bigrams = {text[i : i + 2] for i in range(len(text) - 1)}  # 中文专有名词兜底
        return words | bigrams

    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        q_terms = self._terms(query)
        if not q_terms:
            return chunks[:top_k]
        scored = []
        for c in chunks:
            c_terms = self._terms(c.text)
            overlap = q_terms & c_terms
            score = sum(len(t) for t in overlap) / (sum(len(t) for t in q_terms) or 1)
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, c in scored[:top_k]:
            results.append(
                ScoredChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    source=c.source,
                    heading=c.heading,
                    text=c.text,
                    score=float(score),
                    chunk_index=c.chunk_index,
                    metadata=dict(c.metadata),
                )
            )
        return results


class HttpReranker(BaseReranker):
    """调用 OpenAI 生态之外的 rerank HTTP 接口(DashScope / SiliconFlow 均为 /rerank 端点)。"""

    def __init__(self, base_url: str, api_key: str, model: str):
        import requests

        self._requests = requests
        # self._url = base_url.rstrip("/") + "/rerank"
        self._url = base_url
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._model = model

    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        resp = self._requests.post(
            self._url,
            headers=self._headers,
            json={
                "model": self._model,
                "query": query,
                "documents": [c.text for c in chunks],
                "top_n": top_k,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data["output"]["results"]:
            c = chunks[int(item["index"])]
            results.append(
                ScoredChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    source=c.source,
                    heading=c.heading,
                    text=c.text,
                    score=float(item.get("relevance_score", 0.0)),
                    chunk_index=c.chunk_index,
                    metadata=dict(c.metadata),
                )
            )
        return results


def build_reranker() -> BaseReranker:
    from config import settings

    cfg = settings.get_rerank_config()
    if cfg["provider"] == "overlap":
        return OverlapReranker()
    if not cfg.get("api_key"):
        print(f"[提示] RERANK_PROVIDER={cfg['provider']} 未配置 API Key,回退到本地 OverlapReranker")
        return OverlapReranker()
    print(f"[build_reranker] provider={cfg['provider']}, model={cfg['model']}, base_url={cfg['base_url']}")
    return HttpReranker(base_url=cfg["base_url"], api_key=cfg["api_key"], model=cfg["model"])
