"""工具注册表:Agent 可调用的工具。工具 = 名字 + 自然语言描述 + JSON Schema + 执行函数。

面试点:大模型本身只会"续写文本",工具调用能力来自"把工具清单喂进 prompt,
让模型输出结构化决策 JSON,由外层循环解释执行"——这就是 Agent 的最小闭环。
"""
from __future__ import annotations

import ast
import datetime
import operator
from dataclasses import dataclass
from typing import Any, Callable

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import BaseReranker
from src.vectorstore.store import NumpyVectorStore


@dataclass
class Tool:
    name: str
    description: str  # 给模型看的:什么时候用、干什么
    parameters: dict   # JSON Schema,约束模型生成的参数结构
    func: Callable[[dict], Any]  # 执行函数,接收 args 字典,返回 str(可异步)

    def execute(self, args: dict) -> Any:
        return self.func(args)


def make_kb_search_tool(retriever: HybridRetriever, reranker: BaseReranker | None = None,
                        top_k: int = 4, use_rerank: bool = True) -> Tool:
    def _search(args: dict) -> str:
        query = args.get("query", "")
        chunks = retriever.hybrid_search(query)
        if use_rerank and reranker:
            chunks = reranker.rerank(query, chunks, top_k)
        else:
            chunks = chunks[:top_k]
        if not chunks:
            return "未检索到相关内容"
        lines = [f"- 《{c.source}》({c.heading}) {c.text[:200]}" for c in chunks]
        return "\n".join(lines)

    return Tool(
        name="kb_search",
        description="在企业知识库中检索与问题相关的制度、规范、产品资料。输入为检索词,返回最相关的资料摘录。",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "检索关键词或问题"}},
            "required": ["query"],
        },
        func=_search,
    )


def make_list_documents_tool(store: NumpyVectorStore) -> Tool:
    def _list(_: dict) -> str:
        docs = sorted(set(store.sources))
        return "知识库现有文档:\n" + "\n".join(f"- {d}" for d in docs)

    return Tool(
        name="list_documents",
        description="列出知识库中已有的全部文档清单。当不确定知识库里有没有相关资料时先用它摸底。",
        parameters={"type": "object", "properties": {}},
        func=_list,
    )


# ---- 安全计算器:只允许四则运算,禁 eval ----
_ALLOWED_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_calc(expr: str) -> float:
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](_eval(node.operand))
        raise ValueError(f"不支持的表达式: {ast.dump(node)}")

    return _eval(ast.parse(expr, mode="eval"))


def calculator_tool() -> Tool:
    def _calc(args: dict) -> str:
        expr = str(args.get("expression", "")).strip()
        try:
            return f"{expr} = {_safe_calc(expr)}"
        except Exception as e:  # noqa: BLE001 - 工具要把错误变成字符串反馈给模型
            return f"计算失败:{e}"

    return Tool(
        name="calculator",
        description="安全的四则运算计算器。当需要数值计算(如金额合计、天数换算)时使用。",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "只含数字和 + - * / 的表达式"}},
            "required": ["expression"],
        },
        func=_calc,
    )


def current_time_tool() -> Tool:
    def _now(_: dict) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return Tool(
        name="current_time",
        description="获取当前日期时间。涉及'今天/现在/ deadline'类问题时使用。",
        parameters={"type": "object", "properties": {}},
        func=_now,
    )
