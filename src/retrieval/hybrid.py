"""混合检索:稠密向量 + BM25 稀疏检索,RRF 融合排序。

面试点:
- 为什么融合?向量检索擅长语义近义(“请假”≈“休假”),BM25 擅长精确词匹配;
  两路召回取并集再融合,比任何单路都稳。
- 为什么用 RRF 而不是分数加权?两路分数分布不同(余弦∈[-1,1]、BM25∈[0,+∞)),
  直接加权需要调参和归一化;RRF 只看"排第几",k=60 是论文经典值,开箱即用。
"""
from __future__ import annotations

from config import settings
from src.retrieval.bm25 import BM25Index
from src.vectorstore.store import NumpyVectorStore, ScoredChunk

RRF_K = 60  # RRF 平滑常数,论文经典取值


class HybridRetriever:
    def __init__(self, store: NumpyVectorStore, embedder):
        self.store = store
        self.embedder = embedder
        self.bm25 = BM25Index(store.all_chunks())

    def dense_search(self, query: str, top_k: int) -> list[ScoredChunk]:
        qv = self.embedder.encode([query])[0]
        return self.store.search(qv, top_k=top_k)

    def sparse_search(self, query: str, top_k: int) -> list[ScoredChunk]:
        return self.bm25.search(query, top_k=top_k)

    def hybrid_search(self, query: str, top_k: int | None = None, candidate_k: int | None = None) -> list[ScoredChunk]:
        """两路各取 candidate_k 个候选,RRF 融合后取 top_k。"""
        top_k = top_k or settings.RETRIEVE_TOP_K
        candidate_k = candidate_k or settings.HYBRID_CANDIDATE_K

        dense = self.dense_search(query, candidate_k)
        sparse = self.sparse_search(query, candidate_k)

        rrf: dict[str, float] = {}
        chunk_by_id: dict[str, ScoredChunk] = {}
        for rank, c in enumerate(dense, start=1):
            rrf[c.chunk_id] = rrf.get(c.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            chunk_by_id[c.chunk_id] = c
        for rank, c in enumerate(sparse, start=1):
            rrf[c.chunk_id] = rrf.get(c.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            chunk_by_id.setdefault(c.chunk_id, c)

        fused = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        results = []
        for chunk_id, score in fused:
            c = chunk_by_id[chunk_id]
            results.append(
                ScoredChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    source=c.source,
                    heading=c.heading,
                    text=c.text,
                    score=score,
                    chunk_index=c.chunk_index,
                    metadata=dict(c.metadata),
                )
            )
        return results
