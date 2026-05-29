from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ChunkType = Literal["text", "figure", "table"]
Section = Literal["financial_statements", "mda", "risk_factors", "other"]


@dataclass(frozen=True)
class RawReport:
    path: Path
    company: str
    ticker: str
    year: int
    markdown: str


@dataclass(frozen=True)
class MarkdownBlock:
    block_type: ChunkType
    heading: str
    raw_text: str
    order: int


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    doc_id: str
    company: str
    ticker: str
    year: int
    section: Section
    chunk_type: ChunkType
    seq: int
    raw_text: str
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    context: str | None = None


@dataclass(frozen=True)
class QuestionRecord:
    question_id: str
    chunk_id: str
    question_text: str
    dense_vec: list[float]
    company: str
    ticker: str
    year: int
    section: Section
    chunk_type: ChunkType
