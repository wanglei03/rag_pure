"""Stage 06b:MCP 的 HTTP 传输方式 —— 同一套工具,换一条"网线"。

运行:python stages/stage06_mcp/demo_http.py

与 demo.py(stdio)的差异只在"连接器":
  stdio: server 是子进程,管道即协议,一对一;
  HTTP:  server 是网络服务(streamable-http),多客户端可共享,鉴权/限流都加在这一层。
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.console import Console

from config.settings import BASE_DIR
from src.agent.agent import ToolCallingAgent
from src.llm.base import build_llm
from src.mcp_tool.http_client import McpHttpClient

console = Console()

QUESTION = "帮我查一下去上海出差的住宿报销上限,并按上限 600 元住 3 晚,算出 600*3 最多能报多少?"


def pick_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_port_ready(port: int, deadline: float = 60.0) -> None:
    """HTTP 传输的就绪信号:模块导入 + 知识库加载完成后 uvicorn 才绑端口,
    所以"端口能连通" = "服务已就绪",比 sleep 猜时间可靠。"""
    end = time.time() + deadline
    while time.time() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"等待 MCP HTTP server 端口 {port} 超时")


def main() -> None:
    port = pick_free_port()
    console.print(f"[bold]1. 拉起 HTTP server 子进程[/bold](streamable-http, 端口 {port})")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BASE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    console.print(f"PYTHONPATH: { env["PYTHONPATH"]}, sys.executable: {sys.executable}")
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_tool.mcp_server", "--transport", "streamable-http", "--port", str(port)],
        env=env,
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,   # uvicorn 访问日志走 stdout,关掉保持输出干净
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_port_ready(port)
        console.print("  端口就绪(server 已完成知识库加载)✓")

        console.print("\n[bold]2. HTTP client 连接并握手[/bold]")
        client = McpHttpClient(f"http://127.0.0.1:{port}/mcp")
        console.print(f"  会话 ID(Mcp-Session-Id 头):{client.session_id()}")
        tool_defs = client.list_tool_defs()
        console.print("  发现 MCP 工具:")
        for t in tool_defs:
            console.print(f"    - {t['name']}: {t['description']}")

        console.print("\n[bold]3. 调用工具 + 读取资源[/bold]")
        console.print(f"  list_documents:\n    {client.call_tool('list_documents', {})}")
        stats = client.read_resource("kb://stats")
        stats_text = next((c.text for c in stats.contents if hasattr(c, "text")), "")
        console.print(f"  资源 kb://stats:\n    {stats_text}")

        console.print(f"\n[bold]4. Agent 走 HTTP 工具回答问题:「{QUESTION}」[/bold]")
        agent = ToolCallingAgent(llm=build_llm(), tools=client.to_agent_tools())
        result = agent.run(QUESTION)
        for step in result.steps:
            if step.action == "final":
                console.print(f"  步骤 {step.index}: [green]最终答案[/green]")
            else:
                console.print(f"  步骤 {step.index}: [cyan]{step.action}[/cyan] {step.detail}")
        console.rule("[bold]最终回答[/bold]")
        console.print(result.final_answer)
        console.print("\n[dim]对比:同一套 kb_search/list_documents/calculator 工具,stdio 版见 demo.py;"
                      "传输层切换对 Agent 完全透明。[/dim]")
        client.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
