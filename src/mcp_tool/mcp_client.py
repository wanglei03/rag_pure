"""MCP Client:在独立线程的事件循环里运行 stdio 会话,对 Agent 暴露为同步工具。

设计要点(都是面试可讲的工程决策):
- 异步会话 vs 同步 Agent:客户端独占一个后台线程跑事件循环,用 run_coroutine_threadsafe
  桥接 —— Agent 侧零感知(nested event loop 问题的标准解法);
- 退出必须在同一个 task 里:anyio 的 cancel scope 绑定创建它的 task,
  AsyncExitStack 若跨协程关闭会报 "Attempted to exit cancel scope in a different task"。
  因此把"建立会话 → 待命 → 关闭"放进一个长命协程,用事件通知它收尾;
- 这样写还有一个好处:MCP 工具的调用方完全不关心底层传输(stdio/SSE)与协议细节。
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters, ReadResourceResult
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

from config.settings import BASE_DIR
from src.agent.tools import Tool


class MCPClient:
    """连接一个本地 MCP server 脚本(以 module 方式启动,保证包导入正确)。"""

    def __init__(self, server_module: str = "src.mcp_tool.mcp_server", timeout: float = 60.0):
        self._timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        env = dict(os.environ)
        env["PYTHONPATH"] = str(BASE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        params = StdioServerParameters(command=sys.executable, args=["-m", server_module], env=env)

        self._session: ClientSession | None = None
        self._close_ev: asyncio.Event | None = None
        self._init_error: Exception | None = None
        self._init_done = threading.Event()

        # 长命协程:在同一个 task 里完成 建会话 → 待命 → 清理,规避 cancel scope 跨 task 报错
        self._task = asyncio.run_coroutine_threadsafe(self._run(params), self._loop)
        if not self._init_done.wait(timeout):
            raise TimeoutError(f"MCP server({server_module})连接超时")
        if self._init_error is not None:
            raise RuntimeError(f"MCP server 初始化失败: {self._init_error}")

    async def _run(self, params: StdioServerParameters) -> None:
        self._close_ev = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._session = session
                self._init_done.set()
                await self._close_ev.wait()  # 待命,直到 close() 通知收尾
        except Exception as e:  # noqa: BLE001 - 初始化失败要透传给主线程
            self._init_error = e
            self._init_done.set()

    # ---- 供 Agent 使用的同步接口 ----
    def list_tool_defs(self) -> list[dict]:
        result = asyncio.run_coroutine_threadsafe(self._session.list_tools(), self._loop).result(self._timeout)
        return [
            {"name": t.name, "description": t.description or "", "parameters": t.inputSchema or {"type": "object", "properties": {}}}
            for t in result.tools
        ]

    def call_tool(self, name: str, args: dict) -> str:
        result = asyncio.run_coroutine_threadsafe(self._session.call_tool(name, args), self._loop).result(self._timeout)
        parts = [c.text for c in result.content if hasattr(c, "text")]
        if result.isError:
            return f"MCP 工具 {name} 执行出错: {' '.join(parts)}"
        return "\n".join(parts)

    def read_resource(self, path: str) -> ReadResourceResult:
        return asyncio.run_coroutine_threadsafe(self._session.read_resource(AnyUrl(path)), self._loop).result(self._timeout)

    def to_agent_tools(self) -> list[Tool]:
        """把 server 暴露的全部 MCP 工具包装成本项目的 Tool,直接喂给 Agent。"""
        tools = []
        for spec in self.list_tool_defs():
            name = spec["name"]

            def _make(n: str):
                return lambda args: self.call_tool(n, args)

            tools.append(
                Tool(name=name, description=f"[MCP] {spec['description']}", parameters=spec["parameters"], func=_make(name))
            )
        return tools

    def close(self) -> None:
        if self._close_ev is not None:
            self._loop.call_soon_threadsafe(self._close_ev.set)
            self._task.result(self._timeout)  # 等长命协程(含上下文清理)走完
            self._close_ev = None
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
