"""评估模块:RAG 效果量化 —— 没有评估的 RAG 上线等于裸奔(JD 原话精神的反向表述)。

两类指标:
1. 检索指标(无需 LLM,便宜可批量):hit rate@k、MRR;
2. 生成指标(LLM-as-judge):忠实度(faithfulness)、答案相关性(answer relevancy)。

面试点: how do you know RAG works? —— 先定义指标和评测集,再谈优化;
rerank 到底值不值得加? —— 用 stage07 的对比表说话。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from config import settings
from src.llm.base import BaseLLM
from src.vectorstore.store import ScoredChunk


@dataclass
class EvalItem:
    id: str
    question: str
    gold_doc: str
    gold_chunk_hint: str


def load_eval_set(path: Path | None = None) -> list[EvalItem]:
    path = Path(path or settings.EVAL_SET_PATH)
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EvalItem(**d) for d in data]


# ---------------- 检索指标 ----------------

def hit_rate(gold_doc: str, chunks: list[ScoredChunk]) -> bool:
    """top-k 结果里是否出现了金标文档(按文档粒度,符合真实使用体感)。"""
    return any(c.doc_id == gold_doc for c in chunks)


def reciprocal_rank(gold_doc: str, chunks: list[ScoredChunk]) -> float:
    """第一个命中金标文档的位置的倒数(MRR 的单样本值)。"""
    for rank, c in enumerate(chunks, start=1):
        if c.doc_id == gold_doc:
            return 1.0 / rank
    return 0.0


@dataclass
class RetrievalReport:
    variant: str
    hit_rate_at_k: float
    mrr: float

    def as_row(self) -> tuple[str, str, str]:
        return (self.variant, f"{self.hit_rate_at_k:.2f}", f"{self.mrr:.3f}")


def evaluate_retrieval(items: list[EvalItem], variant_fn, variant: str, top_k: int) -> RetrievalReport:
    """variant_fn(question) -> list[ScoredChunk];对全部题目统计。"""
    hits, rr = 0, 0.0
    for item in items:
        chunks = variant_fn(item.question)[:top_k]
        hits += int(hit_rate(item.gold_doc, chunks))
        rr += reciprocal_rank(item.gold_doc, chunks)
    n = max(len(items), 1)
    return RetrievalReport(variant=variant, hit_rate_at_k=hits / n, mrr=rr / n)


# ---------------- 生成指标(LLM-as-judge) ----------------

FAITHFULNESS_PROMPT = """你是评估评委。给定【资料】和模型【答案】,判断答案中的每个事实性结论是否都能由资料支撑。
只依据资料,不依据你自己的知识。输出 JSON:{{"score": 0到1之间的小数, "reason": "一句话理由"}}

【资料】
{context}

【答案】
{answer}"""

RELEVANCY_PROMPT = """你是评估评委。给定【问题】和模型【答案】,评估答案对问题的针对性和完整程度(1-5 分,5 为完全解答)。
输出 JSON:{{"score": 1到5的整数, "reason": "一句话理由"}}

【问题】
{question}

【答案】
{answer}"""


def _judge(llm: BaseLLM, prompt: str) -> float:
    resp = llm.chat_json([{"role": "user", "content": prompt}])
    return float(resp.get("score", 0.0))


def faithfulness(llm: BaseLLM, context: str, answer: str) -> float:
    return _judge(llm, FAITHFULNESS_PROMPT.format(context=context, answer=answer))


def answer_relevancy(llm: BaseLLM, question: str, answer: str) -> float:
    return _judge(llm, RELEVANCY_PROMPT.format(question=question, answer=answer))


# ---------------- 关键词覆盖率(无需 LLM 的生成质量代理指标) ----------------

def keyword_coverage(answer: str, gold_hint: str) -> float:
    """金标答案里的数字与关键短语在模型答案中出现的比例。便宜、确定、可复现,
    适合作为 LLM-as-judge 的对照(面试点:两套指标互相印证)。"""
    keywords = [k for k in re.split(r"[,,。;;\s]+", gold_hint) if len(k) >= 2]
    if not keywords:
        return 0.0
    hit = sum(1 for k in keywords if k in answer)
    return hit / len(keywords)
