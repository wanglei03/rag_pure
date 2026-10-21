"""Stage 01:文档解析与切分 —— RAG 的第一步,切坏了后面全白费。

运行:python stages/stage01_parsing_chunking/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich import box
from rich.console import Console
from rich.table import Table

from config import settings
from src.ingestion.chunker import chunk_document
from src.ingestion.loaders import load_documents

console = Console()


def main() -> None:
    console.rule("[bold]1. 加载文档[/bold]")
    docs = load_documents(settings.DOCS_DIR)
    table = Table(box=box.SIMPLE_HEAD)
    for col in ["doc_id", "文件", "字符数", "段落数"]:
        table.add_column(col)
    for d in docs:
        table.add_row(d.doc_id, d.source, str(len(d.text)), str(len(d.text.split("\n\n"))))
    console.print(table)

    console.rule("[bold]2. 切分效果(chunk_size 对比)[/bold]")
    target = next(d for d in docs if "考勤" in d.doc_id)
    for size in (160, settings.CHUNK_SIZE, 450):
        chunks = chunk_document(target.doc_id, target.source, target.text, chunk_size=size, overlap=size // 5)
        sizes = [len(c.text) for c in chunks]
        console.print(
            f"chunk_size={size:>3}: 共 {len(chunks)} 块 | 平均 {sum(sizes)//len(sizes)} 字 | "
            f"最小 {min(sizes)} / 最大 {max(sizes)}"
        )

    console.rule(f"[bold]3. 逐块查看(以 {target.source} 为例)[/bold]")
    chunks = chunk_document(target.doc_id, target.source, target.text,
                            chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
    for c in chunks[:3]:
        console.print(f"\n[cyan][{c.chunk_id}][/cyan] 归属标题:《{c.source}》→ {c.heading}")
        console.print(f"  {c.text[:120]}...")


if __name__ == "__main__":
    main()
