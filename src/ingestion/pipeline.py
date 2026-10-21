"""入库管线:文档 → 切分 → 向量化 → 向量库,一键完成并持久化。

stage02 的 demo 直接调用 build_kb();最终应用与 MCP server 也复用它。
"""
from __future__ import annotations

from pathlib import Path

from config import settings
from src.embedding.base import BaseEmbedder
from src.ingestion.chunker import Chunk, chunk_document
from src.ingestion.loaders import load_documents
from src.vectorstore.store import NumpyVectorStore


def ingest_documents(
    docs_dir: Path,
    embedder: BaseEmbedder,
    store: NumpyVectorStore | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> tuple[NumpyVectorStore, list[Chunk]]:
    """返回 (向量库, 全部 chunk)。"""
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    if store is None:
        store = NumpyVectorStore(dim=embedder.dim)

    docs = load_documents(docs_dir)
    if not docs:
        raise RuntimeError(f"{docs_dir} 下没有可解析的文档")

    all_chunks: list[Chunk] = []
    for doc in docs:
        chunks = chunk_document(doc.doc_id, doc.source, doc.text, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        vectors = embedder.encode([c.text for c in chunks])
        store.add(chunks, vectors)
        print(f"[入库] {doc.source}: {len(chunks)} 个 chunk")
    print(f"[完成] 共 {len(all_chunks)} 个 chunk,向量维度 {embedder.dim}")
    return store, all_chunks


def build_kb(force_rebuild: bool = False) -> NumpyVectorStore:
    """供各 stage/应用调用的入口:向量库文件存在则直接加载,否则重建。"""
    settings.ensure_dirs()
    if settings.STORE_PATH.exists() and not force_rebuild:
        return NumpyVectorStore.load(settings.STORE_PATH)

    from src.embedding.base import build_embedder

    embedder = build_embedder()
    store, _ = ingest_documents(settings.DOCS_DIR, embedder)
    store.save(settings.STORE_PATH)
    return store
