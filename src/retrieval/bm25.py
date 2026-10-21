"""稀疏检索:BM25 + jieba 中文分词。

面试点:纯向量检索对"专有名词、编号、条款号"这类精确匹配容易翻车,
BM25 正好互补 —— 这就是混合检索(hybrid)存在的理由。
"""
from __future__ import annotations

import jieba
from rank_bm25 import BM25Okapi

from src.vectorstore.store import ScoredChunk


def tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text.lower()) if t.strip()]


class BM25Index:
    """基于全量 chunk 构建的内存 BM25 索引,检索返回与向量库一致的 ScoredChunk 结构。"""

    def __init__(self, chunks: list[ScoredChunk]):
        self.chunks = chunks
        corpus = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in order:
            c = self.chunks[i]
            results.append(
                ScoredChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    source=c.source,
                    heading=c.heading,
                    text=c.text,
                    score=float(scores[i]),
                    chunk_index=c.chunk_index,
                    metadata=dict(c.metadata),
                )
            )
        return results
