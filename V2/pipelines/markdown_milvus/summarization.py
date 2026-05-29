from __future__ import annotations

import re

from .models import DocumentChunk


def summarize_chunk(
    chunk: DocumentChunk,
    prev_chunk: DocumentChunk | None = None,
    next_chunk: DocumentChunk | None = None,
) -> str:
    if chunk.chunk_type == "table":
        summary = summarize_table(chunk)
    elif chunk.chunk_type == "figure":
        summary = summarize_figure(chunk)
    else:
        summary = summarize_text(chunk)

    neighbor_bits: list[str] = []
    if prev_chunk:
        neighbor_bits.append(f"前序chunk={preview(prev_chunk.raw_text, 36)}")
    if next_chunk:
        neighbor_bits.append(f"后序chunk={preview(next_chunk.raw_text, 36)}")
    if neighbor_bits:
        summary = f"{summary}；上下文：{'；'.join(neighbor_bits)}"
    return summary[:4000]


def summarize_text(chunk: DocumentChunk) -> str:
    first_sentence = re.split(r"[。！？]", chunk.raw_text, maxsplit=1)[0]
    return (
        f"文本摘要：{chunk.company}{chunk.year}年{section_label(chunk.section)}部分提到"
        f"{preview(first_sentence or chunk.raw_text, 120)}"
    )


def summarize_table(chunk: DocumentChunk) -> str:
    lines = [line.strip() for line in chunk.raw_text.splitlines() if line.strip()]
    header = lines[0] if lines else ""
    data_rows = max(0, len(lines) - 2)
    return (
        f"表格摘要：{chunk.company}{chunk.year}年{section_label(chunk.section)}表格，"
        f"表头为{header}，数据行数{data_rows}，内容包括{preview(chunk.raw_text, 160)}"
    )


def summarize_figure(chunk: DocumentChunk) -> str:
    return (
        f"图像摘要：{chunk.company}{chunk.year}年{section_label(chunk.section)}图像资源"
        f"{chunk.raw_text}，用于说明该部分的风险、结构或趋势信息"
    )


def section_label(section: str) -> str:
    labels = {
        "financial_statements": "财务报表",
        "mda": "管理层讨论与分析",
        "risk_factors": "风险因素",
        "other": "其他",
    }
    return labels.get(section, section)


def preview(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."
