"""向量存储:默认 Numpy 实现(零依赖、可持久化),接口按生产向量库设计。

面试点:这里故意用 numpy 暴力检索 —— 数据量小(万级 chunk)时暴力扫完全够用,
且最能讲清"余弦相似度 = 点积(向量已归一化)"的本质。
生产换 Chroma/Milvus/pgvector 时只需实现同一接口,上层代码不变;
差异在于生产库用 HNSW 等近似最近邻索引,把 O(N) 扫描降到 O(log N)。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ScoredChunk:
    chunk_id: str
    doc_id: str
    source: str
    heading: str
    text: str
    score: float
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class NumpyVectorStore:
    """内存向量库:ids + 文本 + 元数据 + 向量矩阵,支持 npz 持久化与元数据过滤。"""

    def __init__(self, dim: int):
        self.dim = dim
        self.ids: list[str] = []
        self.doc_ids: list[str] = []
        self.sources: list[str] = []
        self.headings: list[str] = []
        self.texts: list[str] = []
        self.chunk_indices: list[int] = []
        self.metadatas: list[dict] = []
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    # ---- 写入 ----
    def add(self, chunks: list, vectors: np.ndarray) -> None:
        """chunks: Chunk 列表;vectors: (n, dim) 已归一化"""
        assert len(chunks) == vectors.shape[0], "chunk 数与向量数不一致"
        for c, v in zip(chunks, vectors):
            self.ids.append(c.chunk_id)
            self.doc_ids.append(c.doc_id)
            self.sources.append(c.source)
            self.headings.append(c.heading)
            self.texts.append(c.text)
            self.chunk_indices.append(c.chunk_index)
            # chunk 自身的字段也并入元数据,支持 where={"doc_id": ...} 等检索过滤
            self.metadatas.append(dict(c.metadata) | {"doc_id": c.doc_id, "heading": c.heading})
        self.vectors = np.vstack([self.vectors, vectors]) if len(self.ids) > vectors.shape[0] else vectors.copy()

    # ---- 检索 ----
    def search(self, query_vec: np.ndarray, top_k: int, where: dict | None = None) -> list[ScoredChunk]:
        """暴力余弦检索(向量已归一化 → 点积即余弦)。where: 元数据等值过滤,如 {"doc_id": "..."}"""
        if len(self.ids) == 0:
            return []
        scores = self.vectors @ query_vec  # (N,)
        if where:
            keep = [i for i, m in enumerate(self.metadatas) if all(m.get(k) == v for k, v in where.items())]
            if not keep:
                return []
            order = sorted(keep, key=lambda i: scores[i], reverse=True)[:top_k]
        else:
            order = np.argsort(-scores)[:top_k].tolist()
        return [self._to_scored(i, float(scores[i])) for i in order]

    def _to_scored(self, i: int, score: float) -> ScoredChunk:
        return ScoredChunk(
            chunk_id=self.ids[i],
            doc_id=self.doc_ids[i],
            source=self.sources[i],
            heading=self.headings[i],
            text=self.texts[i],
            score=score,
            chunk_index=self.chunk_indices[i],
            metadata=dict(self.metadatas[i]),
        )

    def all_chunks(self) -> list[ScoredChunk]:
        return [self._to_scored(i, 0.0) for i in range(len(self.ids))]

    # ---- 持久化 ----
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=self.vectors,
            ids=np.array(self.ids, dtype=object),
            doc_ids=np.array(self.doc_ids, dtype=object),
            sources=np.array(self.sources, dtype=object),
            headings=np.array(self.headings, dtype=object),
            texts=np.array(self.texts, dtype=object),
            chunk_indices=np.array(self.chunk_indices, dtype=np.int32),
            metadatas=np.array([json.dumps(m, ensure_ascii=False) for m in self.metadatas], dtype=object),
            dim=np.array([self.dim]),
        )

    @classmethod
    def load(cls, path: Path) -> "NumpyVectorStore":
        data = np.load(Path(path), allow_pickle=True)
        store = cls(dim=int(data["dim"][0]))
        store.vectors = data["vectors"]
        store.ids = list(data["ids"])
        store.doc_ids = list(data["doc_ids"])
        store.sources = list(data["sources"])
        store.headings = list(data["headings"])
        store.texts = list(data["texts"])
        store.chunk_indices = [int(x) for x in data["chunk_indices"]]
        store.metadatas = [json.loads(m) for m in data["metadatas"]]
        return store
