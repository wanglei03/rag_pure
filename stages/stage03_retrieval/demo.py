"""Stage 03:检索 —— 单路召回的局限与混合检索 + 重排的价值。

运行:python stages/stage03_retrieval/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich import box
from rich.console import Console
from rich.table import Table

from config import settings
from src.embedding.base import build_embedder
from src.ingestion.pipeline import build_kb
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import build_reranker

console = Console()

QUERIES = [
    "年假有几天",               # 语义型:向量检索的主场
    "出差结束后 30 天不报销会怎样",  # 精确数字型:BM25 的主场
    "P0 事故响应要求",          # 术语型:两路互补
]


def main() -> None:
    store = build_kb()
    retriever = HybridRetriever(store, build_embedder())
    reranker = build_reranker()

    for q in QUERIES:
        console.rule(f'[bold]查询:「{q}」[/bold]')
        dense = retriever.dense_search(q, settings.RETRIEVE_TOP_K)
        sparse = retriever.sparse_search(q, settings.RETRIEVE_TOP_K)
        hybrid = retriever.hybrid_search(q)
        reranked = reranker.rerank(q, hybrid, settings.RETRIEVE_TOP_K)

        table = Table(box=box.SIMPLE_HEAD, title="各检索方案 top-5(文档 / 得分)")
        table.add_column("排名")
        table.add_column("dense(向量)")
        table.add_column("sparse(BM25)")
        table.add_column("hybrid(RRF)")
        table.add_column("hybrid+rerank")
        for row in range(settings.RETRIEVE_TOP_K):
            def cell(chunks):
                return f"{chunks[row].doc_id[:12]}({chunks[row].score:.2f})" if row < len(chunks) else "-"
            table.add_row(str(row + 1), cell(dense), cell(sparse), cell(hybrid), cell(reranked))
        console.print(table)

    console.rule("[bold]结论[/bold]")
    console.print(
        "1. 向量检索对语义改写(“年假有几天”)稳;BM25 对精确数字(“30 天”)稳;\n"
        "2. RRF 只看排名不看绝对分,两路融合后互补;\n"
        "3. rerank 对候选精排,把最相关的顶到前面 —— 真实提升见 stage07 的指标对比表。"
    )


if __name__ == "__main__":
    main()
