"""文本切分:按段落递归切,保留 markdown 标题作为 chunk 元数据。

面试点:chunk_size 不是越小越好 —— 太小丢上下文,太大引入噪声且塞不进 prompt。
中文文档以 ~200-400 字为宜,并保留 overlap 让边界句子不至于断义。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str  # "{doc_id}#{chunk_index}"
    doc_id: str
    source: str  # 文件名,展示引用时用
    heading: str  # 所属 markdown 标题(最近的上一级)
    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _nearest_heading(text_before: str) -> str:
    """找到 text_before 中最后一个 markdown 标题作为本块的归属标题。"""
    headings = HEADING_RE.findall(text_before)
    return headings[-1][1].strip() if headings else "(文档开头)"


def split_paragraphs(text: str) -> list[str]:
    """双换行分段;过长的单段(无换行的文档)按中文句号再切。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    for p in paras:
        if len(p) <= 400:
            out.append(p)
        else:
            sentences = re.split(r"(?<=[。;!?;])", p)
            buf = ""
            for s in sentences:
                if len(buf) + len(s) > 400 and buf:
                    out.append(buf.strip())
                    buf = s
                else:
                    buf += s
            if buf.strip():
                out.append(buf.strip())
    return out


def recursive_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    """贪心装箱:按段落累加,超过 chunk_size 就封口;overlap 用字符级回退实现。"""
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    chunks: list[str] = []
    buf = ""
    for para in split_paragraphs(text):
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) <= chunk_size:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
        # overlap:从上一块尾部回退 overlap 个字符,保证语义连贯
        tail = buf[-overlap:] if buf else ""
        buf = (tail + "\n" + para) if tail else para
        # 单段仍然超长 → 硬切
        while len(buf) > chunk_size:
            chunks.append(buf[:chunk_size])
            buf = buf[chunk_size - overlap :]
    if buf.strip():
        chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]


def chunk_document(doc_id: str, source: str, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """把一篇文档切成 Chunk 列表,记录归属标题与序号。"""
    # 预计算每个字符位置之前的最近标题,避免每块都全文扫一遍
    headings: list[tuple[int, str]] = []
    for m in HEADING_RE.finditer(text):
        headings.append((m.start(), m.group(2).strip()))

    raw_chunks = recursive_chunk(text, chunk_size=chunk_size, overlap=overlap)
    chunks: list[Chunk] = []
    cursor = 0
    for i, ctext in enumerate(raw_chunks):
        pos = text.find(ctext[: min(20, len(ctext))], cursor)
        if pos == -1:
            pos = cursor
        cursor = pos + len(ctext)
        heading = "(文档开头)"
        for hpos, htext in headings:
            if hpos <= pos:
                heading = htext
            else:
                break
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}#{i}",
                doc_id=doc_id,
                source=source,
                heading=heading,
                text=ctext,
                chunk_index=i,
            )
        )
    return chunks
