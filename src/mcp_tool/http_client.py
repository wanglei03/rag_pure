"""MCP HTTP Client:通过 streamable-http 连接 MCP server,对 Agent 暴露为同步工具。

与 stdio 版客户端(MCPClient)同构:后台线程跑事件循环 + run_coroutine_threadsafe 桥接
+ 长命协程保证退出清理在同一 task。差异只在"连接器":

- stdio:   stdio_client(StdioServerParameters) —— server 是子进程,管道即协议;
- HTTP:    streamable_http_client(url, http_client) —— server 是网络服务,HTTP + SSE 流式响应;
  headers 不再由连接器直接接收,而是预先构造 httpx.AsyncClient(create_mcp_http_client)。

HTTP 传输独有的点(面试可讲):
- 会话:streamable-http 用 Mcp-Session-Id 请求头维持会话,server 侧可设 max_sessions /
  session_idle_timeout 做资源治理;初始化想完全无状态可开 stateless_http(牺牲断线续传);
- 鉴权:客户端走 headers 注入 token,server 侧 FastMCP(token_verifier=...) 或反向代理统一鉴权。
  政企私有化交付里,鉴权层放反代还是放 server,是常见的架构讨论点;
- 就绪信号:HTTP 下"端口能连通"≈"服务已就绪"(模块导入/知识库加载完成后 uvicorn 才绑端口),
  比 stdio 的 spawn 即返回更可靠。
"""
from __future__ import annotations

import asyncio
import threading
from contextlib import AsyncExitStack
from typing import Callable

from mcp import ClientSession, ReadResourceResult
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from pydantic import AnyUrl

from src.agent.tools import Tool


class McpHttpClient:
    """连接一个 streamable-http MCP server,接口与 stdio 版 MCPClient 完全一致。"""

    def __init__(self, url: str = "http://127.0.0.1:8000/mcp",
                 headers: dict[str, str] | None = None, timeout: float = 60.0):
        self._timeout = timeout
        self._url = url
        self._headers = headers
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        self._session: ClientSession | None = None
        self._close_ev: asyncio.Event | None = None
        self._init_error: Exception | None = None
        self._init_done = threading.Event()
        self._get_session_id: Callable[[], str | None] | None = None

        self._task = asyncio.run_coroutine_threadsafe(self._run(), self._loop)
        if not self._init_done.wait(timeout):
            raise TimeoutError(f"MCP HTTP server({url})连接超时")
        if self._init_error is not None:
            raise RuntimeError(f"MCP HTTP server 初始化失败: {self._init_error}")

    async def _run(self) -> None:
        self._close_ev = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                # 自建的 httpx client 生命周期归调用方,一并交给 ExitStack 管理
                http_client = await stack.enter_async_context(create_mcp_http_client(headers=self._headers))
                read, write, self._get_session_id = await stack.enter_async_context(
                    streamable_http_client(self._url, http_client=http_client)
                )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._session = session
                self._init_done.set()
                await self._close_ev.wait()  # 待命,直到 close() 通知收尾
        except Exception as e:  # noqa: BLE001 - 初始化失败要透传给主线程
            self._init_error = e
            self._init_done.set()

    # ---- 供 Agent 使用的同步接口(与 stdio 版一致) ----
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

    def session_id(self) -> str | None:
        """streamable-http 的会话标识;server 靠这个请求头识别会话。"""
        if self._get_session_id is None:
            return None
        fut = asyncio.run_coroutine_threadsafe(_awaitable(self._get_session_id), self._loop)
        return fut.result(self._timeout)

    def to_agent_tools(self) -> list[Tool]:
        """把 server 暴露的全部 MCP 工具包装成本项目的 Tool,直接喂给 Agent。"""
        tools = []
        for spec in self.list_tool_defs():
            name = spec["name"]

            def _make(n: str):
                return lambda args: self.call_tool(n, args)

            tools.append(
                Tool(name=name, description=f"[MCP-HTTP] {spec['description']}", parameters=spec["parameters"], func=_make(name))
            )
        return tools

    def close(self) -> None:
        if self._close_ev is not None:
            self._loop.call_soon_threadsafe(self._close_ev.set)
            self._task.result(self._timeout)
            self._close_ev = None
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)


async def _awaitable(fn):
    """session_id 可能是同步 callable,统一包成协程在线程事件循环里执行。"""
    return fn()
