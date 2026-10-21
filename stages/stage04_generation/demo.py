"""Stage 04:生成 —— 让模型"只根据资料说话",并给出可核查的引用。

运行:python stages/stage04_generation/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.console import Console

from src.generation.qa_chain import QAChain

console = Console()


def main() -> None:
    chain = QAChain()

    console.rule('[bold]问题 1(可回答):上海出差住宿报销上限?[/bold]')
    r1 = chain.ask("去上海出差,住宿费报销上限是多少钱一晚?")
    console.print(f"[green]回答:[/green]{r1.answer}")
    console.print("[dim]引用来源:[/dim]")
    for c in r1.cited:
        console.print(f"  [{c.chunk_id}] 《{c.source}》{c.heading}")

    console.rule("[bold]问题 2(资料外问题 → 应拒答):食堂在哪一层?[/bold]")
    r2 = chain.ask("公司食堂在几楼,午餐时间是什么?")
    console.print(f"[yellow]回答:[/yellow]{r2.answer}")
    console.print(f"[dim]是否拒答:{'是' if r2.refused else '否'}(资料库确实没有这类信息)[/dim]")

    console.rule("[bold]问题 3(开启/关闭 rerank 的引用对比)[/bold]")
    r3_off = chain.ask("P0 事故要求多久内响应?", use_rerank=False)
    r3_on = chain.ask("P0 事故要求多久内响应?", use_rerank=True)
    console.print(f"  关 rerank 首条引用:{r3_off.retrieved[0].chunk_id} 《{r3_off.retrieved[0].source}》")
    console.print(f"  开 rerank 首条引用:{r3_on.retrieved[0].chunk_id} 《{r3_on.retrieved[0].source}》")


if __name__ == "__main__":
    main()
