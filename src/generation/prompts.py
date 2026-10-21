"""生成层:带引用约束的 grounded prompt + 引用解析。

面试点:抑制幻觉的三个工程手段——
1. 指令约束:只允许依据资料回答,资料不足必须说不知道;
2. 引用标注:每个结论带来源编号,用户可核查,也让"忠实度"可以被自动评估;
3. 引用回收:解析答案里的 [n],只把真正被引用的 chunk 展示给用户(没引用的召回 chunk 不展示)。
"""
from __future__ import annotations

import re

RAG_SYSTEM = """你是企业知识库问答助手。请严格遵守:
1. 只能根据用户提供的【资料】回答问题,不得使用资料之外的知识;
2. 每个结论后面用 [编号] 标注来源,如 [1];
3. 如果资料不足以回答问题,直接回答"根据现有资料无法回答",并建议补充什么资料;
4. 回答使用简体中文,简洁准确。"""

RAG_USER_TEMPLATE = """【资料】
{context}

问题:{question}"""


def build_context(chunks) -> str:
    """把召回 chunk 编号组装进 prompt,编号与最终展示一致。"""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] 《{c.source}》({c.heading})\n{c.text}")
    return "\n\n".join(blocks)


def build_rag_messages(question: str, chunks) -> list[dict]:
    return [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": RAG_USER_TEMPLATE.format(context=build_context(chunks), question=question)},
    ]


CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def parse_citations(answer: str, n_chunks: int) -> set[int]:
    """解析答案中引用到的资料编号(1 起),过滤掉超出范围的编号。"""
    cited = {int(m) for m in CITATION_RE.findall(answer)}
    return {i for i in cited if 1 <= i <= n_chunks}


def cited_chunks(answer: str, chunks) -> list:
    """只返回被答案实际引用的 chunk,保持引用编号与展示顺序一致。"""
    idx = sorted(parse_citations(answer, len(chunks)))
    return [chunks[i - 1] for i in idx]
