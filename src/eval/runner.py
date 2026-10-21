"""评估执行器:一次跑完 检索对比 + 生成质量,输出汇总表。"""
from __future__ import annotations

from config import settings
from src.eval.metrics import (
    EvalItem,
    RetrievalReport,
    answer_relevancy,
    evaluate_retrieval,
    faithfulness,
    keyword_coverage,
    load_eval_set,
)
from src.generation.prompts import build_context
from src.generation.qa_chain import QAChain
from src.llm.base import build_llm


def compare_retrieval_variants(items: list[EvalItem], chain: QAChain, top_k: int | None = None) -> list[RetrievalReport]:
    top_k = top_k or settings.RETRIEVE_TOP_K
    r = chain.retriever
    reranker = chain.reranker
    reports = []
    reports.append(
        evaluate_retrieval(items, lambda q: r.dense_search(q, settings.HYBRID_CANDIDATE_K), "dense(纯向量)", top_k)
    )
    reports.append(
        evaluate_retrieval(items, lambda q: r.sparse_search(q, settings.HYBRID_CANDIDATE_K), "sparse(纯BM25)", top_k)
    )
    reports.append(
        evaluate_retrieval(items, lambda q: r.hybrid_search(q), "hybrid(RRF融合)", top_k)
    )
    if reranker:
        reports.append(
            evaluate_retrieval(
                items,
                lambda q: reranker.rerank(q, r.hybrid_search(q, candidate_k=settings.HYBRID_CANDIDATE_K), top_k),
                "hybrid+rerank(完整链路)",
                top_k,
            )
        )
    return reports


def evaluate_generation(items: list[EvalItem], chain: QAChain) -> list[dict]:
    llm = build_llm()
    rows = []
    for item in items:
        result = chain.ask(item.question)
        context = build_context(result.cited or result.retrieved[: settings.RETRIEVE_TOP_K])
        rows.append(
            {
                "id": item.id,
                "question": item.question,
                "refused": result.refused,
                "hit": any(c.doc_id == item.gold_doc for c in result.retrieved),
                "faithfulness": faithfulness(llm, context, result.answer),
                "relevancy": answer_relevancy(llm, item.question, result.answer),
                "keyword_coverage": keyword_coverage(result.answer, item.gold_chunk_hint),
            }
        )
    return rows


def run_full_eval() -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()
    items: list[EvalItem] = load_eval_set()
    chain = QAChain()

    console.rule("[bold]检索效果对比(hit rate@k / MRR)[/bold]")
    table = Table(box=box.SIMPLE_HEAD)
    for col in ["检索方案", "Hit Rate", "MRR"]:
        table.add_column(col, justify="left")
    for report in compare_retrieval_variants(items, chain):
        table.add_row(*report.as_row())
    console.print(table)

    console.rule("[bold]生成质量(LLM-as-judge + 关键词覆盖)[/bold]")
    gen_table = Table(box=box.SIMPLE_HEAD)
    for col in ["题号", "命中", "拒答", "忠实度", "相关性", "关键词覆盖"]:
        gen_table.add_column(col, justify="left")
    rows = evaluate_generation(items, chain)
    for r in rows:
        gen_table.add_row(
            r["id"],
            "✓" if r["hit"] else "✗",
            "是" if r["refused"] else "否",
            f"{r['faithfulness']:.2f}",
            f"{r['relevancy']:.1f}",
            f"{r['keyword_coverage']:.2f}",
        )
    n = max(len(rows), 1)
    gen_table.add_row(
        "均值", "", "",
        f"{sum(r['faithfulness'] for r in rows) / n:.2f}",
        f"{sum(r['relevancy'] for r in rows) / n:.1f}",
        f"{sum(r['keyword_coverage'] for r in rows) / n:.2f}",
    )
    console.print(gen_table)
    console.print("[dim]mock 模式下忠实度/相关性为固定评分,仅供演示流程;"
                  "切到真实模型后即为真实指标。[/dim]")
