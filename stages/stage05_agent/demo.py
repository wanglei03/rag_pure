"""Stage 05:Agent —— 让模型自己决定"查什么、什么时候算、什么时候答"。

运行:python stages/stage05_agent/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.console import Console

from src.agent.agent import ToolCallingAgent
from src.agent.tools import calculator_tool, current_time_tool, make_kb_search_tool, make_list_documents_tool
from src.embedding.base import build_embedder
from src.ingestion.pipeline import build_kb
from src.llm.base import build_llm
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import build_reranker

console = Console()

QUESTION = "帮我查一下去上海出差的住宿报销上限,并按上限 600 元住 3 晚,算出 600*3 最多能报多少?"


def main() -> None:
    console.print(f"[bold]用户问题:[/bold]{QUESTION}\n")

    store = build_kb()
    retriever = HybridRetriever(store, build_embedder())
    reranker = build_reranker()
    agent = ToolCallingAgent(
        llm=build_llm(),
        tools=[
            make_kb_search_tool(retriever, reranker),
            calculator_tool(),
            make_list_documents_tool(store),
            current_time_tool(),
        ],
    )

    result = agent.run(QUESTION)

    console.rule("[bold]Agent 执行轨迹[/bold]")
    for step in result.steps:
        if step.action == "final":
            console.print(f"  步骤 {step.index}: [green]输出最终答案[/green]")
            console.print(f"    {step.detail}")
        else:
            console.print(f"  步骤 {step.index}: [cyan]调用 {step.action}[/cyan] 参数 {step.detail}")
            console.print(f"    [dim]观测:{step.observation[:160]}[/dim]")

    console.rule("[bold]最终回答[/bold]")
    console.print(result.final_answer)


if __name__ == "__main__":
    main()
