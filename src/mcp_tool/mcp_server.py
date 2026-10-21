"""MCP Server:把知识库检索封装成标准 MCP 服务,支持两种传输:

  stdio(默认):  python -m src.mcp_tool.mcp_server
      本地工具场景,子进程方式启动,不需要端口。

  streamable-http: python -m src.mcp_tool.mcp_server --transport streamable-http --port 8000
      远程服务场景,多客户端共享,端点 http://127.0.0.1:8000/mcp
      (鉴权切入点:FastMCP 的 token_verifier 参数 / 反向代理层的 auth)。

两个传输复用同一套工具定义,差别只在 run() 的 transport 参数 —— 这就是
"能力定义与传输解耦"的意思,也是 JD 里 MCP 的工程意义:知识库从"某个项目内部
的函数"变成"网络上可发现、可复用的标准能力"。

三个工程要点(都是真实踩坑记录,面试可讲):
1. stdio 传输下 stdout 是协议通道,任何 print 都会污染协议(客户端 JSONRPC 解析直接失败)——
   加载期的所有 print 一律改道 stderr,既保住协议干净,又不丢 provider/model 这类排障信息;
2. numpy/BLAS 类库"禁止"在 MCP 事件循环的异步 handler 里首次 import
   (Windows 上 OpenBLAS/MKL 线程初始化与 anyio/uvicorn 工作线程会死锁,
   症状是 tools/call 永不返回)。HTTP 传输同样成立 —— uvicorn 的 handler 也在事件循环里跑。
   解法:模块导入时就饥渴加载知识库,把重活全部挡在 mcp.run() 启动事件循环之前;
3. HTTP 传输下"端口就绪 ≠ 服务就绪":uvicorn 监听端口前要先完成模块导入(含知识库加载),
   所以 demo 里"端口能连上"就可以当作 server 就绪信号,不用 sleep 猜时间。
"""
from __future__ import annotations

import contextlib
import sys

from mcp.server.fastmcp import FastMCP

from src.generation.qa_chain import _get_embedder
from src.ingestion.pipeline import build_kb
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import build_reranker


def _load_kb():
    # 整个加载过程(入库 + embedder + reranker)的 print 全部改道 stderr:
    # stdout 只能留给 JSONRPC,漏一句日志客户端就会解析失败。
    with contextlib.redirect_stdout(sys.stderr):
        store = build_kb()
        retriever = HybridRetriever(store, _get_embedder())
        reranker = build_reranker()
    return retriever, reranker


# 饥渴加载:必须在 mcp.run() 启动事件循环之前完成 numpy/jieba 的首次导入
_RETRIEVER, _RERANKER = _load_kb()


def build_server(host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    """构造 server 实例并注册全部工具。host/port 只对 HTTP 传输生效。"""
    mcp = FastMCP("enterprise-kb", host=host, port=port)

    @mcp.tool()
    async def kb_search(query: str) -> str:
        """在企业知识库中检索相关制度、规范、产品资料,返回带出处编号的摘录。"""
        chunks = _RETRIEVER.hybrid_search(query)
        chunks = _RERANKER.rerank(query, chunks, 4)
        lines = []
        for i, c in enumerate(chunks, start=1):
            lines.append(f"[{i}] 《{c.source}》{c.heading}\n{c.text[:300]}")
        return "\n\n".join(lines) if lines else "未检索到相关内容"

    @mcp.tool()
    async def list_documents() -> str:
        """列出知识库中已有的全部文档。"""
        docs = sorted(set(_RETRIEVER.store.sources))
        return "\n".join(f"- {d}" for d in docs)

    @mcp.tool()
    async def calculator(expression: str) -> str:
        """安全的四则运算计算器,用于金额合计、天数换算等数值问题。"""
        from src.agent.tools import _safe_calc

        try:
            return f"{expression} = {_safe_calc(expression)}"
        except Exception as e:  # noqa: BLE001
            return f"计算失败:{e}"

    @mcp.resource("kb://stats")
    async def stats() -> str:
        """知识库统计信息。"""
        store = _RETRIEVER.store
        from collections import Counter

        per_doc = Counter(store.doc_ids)
        lines = [f"chunk 总数: {len(store.ids)}", f"向量维度: {store.dim}"]
        lines += [f"- {doc}: {n} chunks" for doc, n in sorted(per_doc.items())]
        return "\n".join(lines)

    return mcp


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="企业知识库 MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = build_server(host=args.host, port=args.port)
    if args.transport == "streamable-http":
        print(f"[mcp_tool] HTTP 端点: http://{args.host}:{args.port}/mcp(按 Ctrl+C 停止)", file=sys.stderr)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
