"""Stage 06:MCP —— 知识库不再是一个函数,而是一个标准服务。

运行:python stages/stage06_mcp/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.console import Console

from src.agent.agent import ToolCallingAgent
from src.llm.base import build_llm
from src.mcp_tool.mcp_client import MCPClient

console = Console()

QUESTION = "帮我查一下去上海出差的住宿报销上限,并按上限 600 元住 3 晚,算出 600*3 最多能报多少?"


def main() -> None:
    console.print("[bold]1. 连接 MCP server(stdio)…[/bold]")
    client = MCPClient("src.mcp_tool.mcp_server")

    tool_defs = client.list_tool_defs()
    console.print("  发现 MCP 工具:")
    for t in tool_defs:
        console.print(f"    - {t['name']}: {t['description']} : {t['parameters']}")

    console.print("\n[bold]2. 资源探查[/bold]")
    console.print(f"  \ncall tool list_documents:\n    {client.call_tool('list_documents', {})}")
    console.print(f"  \nkb://stats:\n    {client.read_resource('kb://stats')}")

    console.print(f"\n[bold]3. Agent 使用 MCP 工具回答问题:「{QUESTION}」[/bold]")
    agent = ToolCallingAgent(llm=build_llm(), tools=client.to_agent_tools())
    result = agent.run(QUESTION)
    for step in result.steps:
        if step.action == "final":
            console.print(f"  步骤 {step.index}: [green]最终答案[/green] {step.detail}")
        else:
            console.print(f"  步骤 {step.index}: [cyan]{step.action}[/cyan] {step.detail}")

    console.rule("[bold]最终回答[/bold]")
    console.print(result.final_answer)
    console.print("\n[dim]注意:Agent 的工具列表来自 MCP server,本项目对 MCP 协议零魔改 —— "
                  "同一 server 可直接挂到 Claude Desktop / Qoder 等任意 MCP 客户端。[/dim]")
    client.close()


if __name__ == "__main__":
    main()
