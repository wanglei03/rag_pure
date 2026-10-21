"""QA 链:检索 → (可选)重排 → 生成 → 引用回收,一次调用拿到带溯源的答案。"""
from __future__ import annotations

from dataclasses import dataclass, field

from config import settings
from src.generation.prompts import build_rag_messages, cited_chunks
from src.ingestion.pipeline import build_kb
from src.llm.base import BaseLLM, build_llm
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import BaseReranker, build_reranker
from src.vectorstore.store import ScoredChunk


@dataclass
class QAResult:
    question: str
    answer: str
    retrieved: list[ScoredChunk] = field(default_factory=list)  # 召回的候选
    cited: list[ScoredChunk] = field(default_factory=list)      # 答案实际引用的
    refused: bool = False  # 模型是否表示"资料不足"


class QAChain:
    def __init__(self, llm: BaseLLM | None = None, retriever: HybridRetriever | None = None,
                 reranker: BaseReranker | None = None, top_k: int | None = None):
        store = build_kb()
        self.llm = llm or build_llm()
        self.retriever = retriever or HybridRetriever(store, _get_embedder())
        self.reranker = reranker if reranker is not None else build_reranker()
        self.top_k = top_k or settings.RETRIEVE_TOP_K

    def ask(self, question: str, use_rerank: bool = True) -> QAResult:
        candidates = self.retriever.hybrid_search(question, candidate_k=max(self.top_k * 4, settings.HYBRID_CANDIDATE_K))
        if use_rerank and self.reranker:
            candidates = self.reranker.rerank(question, candidates, self.top_k)
        else:
            candidates = candidates[: self.top_k]

        messages = build_rag_messages(question, candidates)
        answer = self.llm.chat(messages)
        cited = cited_chunks(answer, candidates)
        refused = "无法回答" in answer
        return QAResult(question=question, answer=answer, retrieved=candidates, cited=cited, refused=refused)


_EMBEDDER = None


def _get_embedder():
    """QAChain 与入库用同一个 embedder,避免 mock/真实向量空间不一致。"""
    global _EMBEDDER
    if _EMBEDDER is None:
        from src.embedding.base import build_embedder

        _EMBEDDER = build_embedder()
    return _EMBEDDER
