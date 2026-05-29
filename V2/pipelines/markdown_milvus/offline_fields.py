from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from .models import DocumentChunk, MarkdownBlock, RawReport, Section


def make_doc_id(ticker: str, year: int) -> str:
    return f"{ticker}_{year}"


def make_chunk_id(ticker: str, year: int, seq: int) -> str:
    return f"{ticker}_{year}_{seq:04d}"


def derive_section(heading: str) -> Section:
    lowered = heading.lower()
    if any(keyword in lowered for keyword in ("风险", "risk")):
        return "risk_factors"
    if any(keyword in lowered for keyword in ("财务", "现金流", "资产负债", "利润", "financial")):
        return "financial_statements"
    if any(keyword in lowered for keyword in ("管理层", "经营", "业务", "mda", "discussion", "analysis")):
        return "mda"
    return "other"


def assign_chunk_identity(report: RawReport, blocks: list[MarkdownBlock]) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    doc_id = make_doc_id(report.ticker, report.year)
    for seq, block in enumerate(blocks, start=1):
        chunks.append(
            DocumentChunk(
                chunk_id=make_chunk_id(report.ticker, report.year, seq),
                doc_id=doc_id,
                company=report.company,
                ticker=report.ticker,
                year=report.year,
                section=derive_section(block.heading),
                chunk_type=block.block_type,
                seq=seq,
                raw_text=block.raw_text,
            )
        )
    return chunks


def link_neighbors(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    grouped: dict[str, list[DocumentChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)

    linked: list[DocumentChunk] = []
    for doc_chunks in grouped.values():
        ordered = sorted(doc_chunks, key=lambda item: item.seq)
        for index, chunk in enumerate(ordered):
            prev_id = ordered[index - 1].chunk_id if index > 0 else None
            next_id = ordered[index + 1].chunk_id if index + 1 < len(ordered) else None
            linked.append(replace(chunk, prev_chunk_id=prev_id, next_chunk_id=next_id))

    return sorted(linked, key=lambda item: (item.doc_id, item.seq))


def attach_context(
    chunks: list[DocumentChunk],
    summarizer: Callable[[DocumentChunk, DocumentChunk | None, DocumentChunk | None], str],
) -> list[DocumentChunk]:
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    contextualized: list[DocumentChunk] = []
    for chunk in chunks:
        prev_chunk = by_id.get(chunk.prev_chunk_id or "")
        next_chunk = by_id.get(chunk.next_chunk_id or "")
        contextualized.append(replace(chunk, context=summarizer(chunk, prev_chunk, next_chunk)))
    return contextualized
