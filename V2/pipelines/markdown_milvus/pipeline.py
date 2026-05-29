from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pymilvus import MilvusClient

from .embeddings import EmbeddingProvider, HashEmbeddingProvider
from .markdown_parser import load_raw_reports, parse_markdown_report
from .milvus_store import insert_documents, insert_questions, reset_database
from .models import DocumentChunk, QuestionRecord
from .offline_fields import assign_chunk_identity, attach_context, link_neighbors
from .question_generation import build_question_records
from .summarization import summarize_chunk


@dataclass(frozen=True)
class IndexBuildResult:
    db_path: Path
    raw_reports_dir: Path
    chunks: list[DocumentChunk]
    questions: list[QuestionRecord]


def build_document_chunks(raw_reports_dir: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    reports = load_raw_reports(raw_reports_dir)
    if not reports:
        raise RuntimeError(f"No markdown reports found under {raw_reports_dir}")

    for report in reports:
        blocks = parse_markdown_report(report)
        chunks.extend(assign_chunk_identity(report, blocks))

    linked = link_neighbors(chunks)
    return attach_context(linked, summarize_chunk)


def build_questions(
    chunks: list[DocumentChunk],
    embedding_provider: EmbeddingProvider | None = None,
) -> list[QuestionRecord]:
    provider = embedding_provider or HashEmbeddingProvider()
    return build_question_records(chunks, provider)


def build_milvus_index(
    raw_reports_dir: Path,
    db_path: Path,
    embedding_provider: EmbeddingProvider | None = None,
) -> tuple[MilvusClient, IndexBuildResult]:
    chunks = build_document_chunks(raw_reports_dir)
    questions = build_questions(chunks, embedding_provider)
    client = reset_database(db_path)
    insert_documents(client, chunks)
    insert_questions(client, questions)
    return client, IndexBuildResult(
        db_path=db_path,
        raw_reports_dir=raw_reports_dir,
        chunks=chunks,
        questions=questions,
    )
