"""Stage 02:向量化与存储 —— 文本如何变成可计算的相似度。

运行:python stages/stage02_embedding_vectorstore/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from rich.console import Console

from config import settings
from src.embedding.base import build_embedder
from src.ingestion.pipeline import build_kb

console = Console()


def main() -> None:
    embedder = build_embedder()
    console.print(f"[bold]当前 embedding:[/bold] {embedder.name if hasattr(embedder, 'name') else type(embedder).__name__}"
                  f"(dim={embedder.dim})")

    console.rule("[bold]1. 相似度直觉:三句话两两余弦相似度[/bold]")
    sents = [
        "员工年假按工龄核定,满 3 年有 7 天",
        "年假天数与工龄挂钩,三年以上十年以下 7 天",
        "数据库变更必须提前提交变更单",
    ]
    vecs = embedder.encode(sents)
    for i in range(len(sents)):
        for j in range(i + 1, len(sents)):
            sim = float(vecs[i] @ vecs[j])  # 已 L2 归一化 → 点积即余弦
            console.print(f"  cos(句{i+1}, 句{j+1}) = {sim:+.3f}   {sents[i][:14]}… ↔ {sents[j][:14]}…")

    console.rule("[bold]2. 建库 + 持久化 + 重新加载[/bold]")
    store = build_kb(force_rebuild=True)
    console.print(f"  已保存到 {settings.STORE_PATH.name},重新加载校验:")
    store2 = type(store).load(settings.STORE_PATH)
    assert np.allclose(store.vectors, store2.vectors), "持久化数据不一致!"
    console.print(f"  加载后 chunk 数 = {len(store2.ids)},与内存一致 ✓")

    console.rule("[bold]3. 向量检索 + 元数据过滤[/bold]")
    query = "出差住宿能报多少钱"
    qv = embedder.encode([query])[0]
    console.print(f"  查询:「{query}」")
    for c in store.search(qv, top_k=3):
        console.print(f"    {c.score:.3f} | 《{c.source}》{c.heading} | {c.text[:40]}…")

    console.print("  只搜「差旅与费用报销制度」:")
    for c in store.search(qv, top_k=2, where={"doc_id": "差旅与费用报销制度"}):
        console.print(f"    {c.score:.3f} | 《{c.source}》{c.heading} | {c.text[:40]}…")


if __name__ == "__main__":
    main()
