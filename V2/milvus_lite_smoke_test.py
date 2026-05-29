#!/usr/bin/env python
"""Smoke validation for the V2 Milvus Lite markdown index pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    from pipelines.markdown_milvus.embeddings import HashEmbeddingProvider
    from pipelines.markdown_milvus.milvus_store import (
        fetch_documents_by_chunk_ids,
        get_collection_count,
        hybrid_search_questions,
        question_hits_to_rows,
    )
    from pipelines.markdown_milvus.models import DocumentChunk, QuestionRecord
    from pipelines.markdown_milvus.pipeline import build_milvus_index
    from pipelines.markdown_milvus.schema import DOCUMENT_COLLECTION_NAME, QUESTION_COLLECTION_NAME
    from pipelines.markdown_milvus.summarization import preview
except ImportError as exc:
    print(
        "Milvus V2 smoke dependencies are missing. Install requirements first:\n"
        "  /root/miniconda3/bin/python -m pip install -r V2/requirements.txt\n"
        f"Import error: {exc}"
    )
    sys.exit(1)


ROOT_DIR = Path(__file__).resolve().parent
RAW_REPORTS_DIR = ROOT_DIR / "data" / "raw_reports"
DB_PATH = ROOT_DIR / "milvus_lite_smoke.db"


def run_scenario(
    client: Any,
    embedding_provider: HashEmbeddingProvider,
    title: str,
    query: str,
    filter_expr: str,
    expected_chunk_type: str,
) -> list[dict[str, Any]]:
    hits = hybrid_search_questions(
        client=client,
        query_text=query,
        query_vector=embedding_provider.embed_text(query),
        limit=3,
        filter_expr=filter_expr,
    )
    rows = question_hits_to_rows(hits)
    documents = fetch_documents_by_chunk_ids(client, [row["chunk_id"] for row in rows])

    print(f"\n{'=' * 88}")
    print(title)
    print(f"Query: {query}")
    print(f"Filter: {filter_expr}")
    print(f"Question hits: {len(rows)}")
    print("-" * 88)

    if not rows:
        raise AssertionError(f"{title} returned no question hits")

    for rank, row in enumerate(rows, start=1):
        document = documents.get(row["chunk_id"])
        if not document:
            raise AssertionError(f"Missing document for chunk_id={row['chunk_id']}")
        if row["chunk_type"] != expected_chunk_type:
            raise AssertionError(
                f"{title} expected chunk_type={expected_chunk_type}, got {row['chunk_type']}"
            )
        if not document.get("raw_text"):
            raise AssertionError(f"Document {row['chunk_id']} has empty raw_text")

        print(
            f"#{rank} score={row['score']:.4f} chunk_id={row['chunk_id']} "
            f"type={row['chunk_type']} section={row['section']}"
        )
        print(f"   question: {preview(row['question_text'], 120)}")
        print(f"   raw_text: {preview(document['raw_text'], 120)}")
        print(f"   context: {preview(document.get('context') or '', 140)}")
        print(
            f"   prev={document.get('prev_chunk_id')} "
            f"next={document.get('next_chunk_id')}"
        )

    return rows


def print_data_summary(chunks: list[DocumentChunk], questions: list[QuestionRecord]) -> None:
    chunk_types = sorted({chunk.chunk_type for chunk in chunks})
    question_multiplier_ok = len(chunks) * 3 <= len(questions) <= len(chunks) * 5
    year_is_int = all(isinstance(chunk.year, int) for chunk in chunks)

    print(f"Created collections: {DOCUMENT_COLLECTION_NAME}, {QUESTION_COLLECTION_NAME}")
    print(f"Raw markdown dir: {RAW_REPORTS_DIR}")
    print(f"Document chunks: {len(chunks)}")
    print(f"Question records: {len(questions)}")
    print(f"Chunk types: {', '.join(chunk_types)}")
    print(f"Questions per chunk in [3, 5]: {question_multiplier_ok}")
    print(f"Year values are int: {year_is_int}")

    if not question_multiplier_ok:
        raise AssertionError("Question count is not within 3-5 records per chunk")
    if not year_is_int:
        raise AssertionError("At least one chunk year is not int")


def main() -> None:
    embedding_provider = HashEmbeddingProvider()
    client, result = build_milvus_index(
        raw_reports_dir=RAW_REPORTS_DIR,
        db_path=DB_PATH,
        embedding_provider=embedding_provider,
    )
    try:
        print_data_summary(result.chunks, result.questions)
        print(f"Milvus document rows: {get_collection_count(client, DOCUMENT_COLLECTION_NAME, 'chunk_id')}")
        print(f"Milvus question rows: {get_collection_count(client, QUESTION_COLLECTION_NAME, 'question_id')}")

        run_scenario(
            client,
            embedding_provider,
            "1) Text query over markdown paragraph chunks",
            "宁德时代2024年研发和动力电池技术有什么进展？",
            'ticker == "CATL" and year == 2024 and chunk_type == "text"',
            "text",
        )
        run_scenario(
            client,
            embedding_provider,
            "2) Table query over markdown table chunks",
            "宁德时代2024年现金流表格披露了什么？",
            'ticker == "CATL" and year == 2024 and chunk_type == "table"',
            "table",
        )
        run_scenario(
            client,
            embedding_provider,
            "3) Figure query over markdown image chunks",
            "比亚迪2024年风险图展示了哪些风险？",
            'ticker == "BYD" and year == 2024 and chunk_type == "figure"',
            "figure",
        )

        print(f"\n{'=' * 88}")
        print("Smoke validation passed")
        print(f"- Milvus Lite DB path: {DB_PATH}")
        print("- Retrieval flow: question_collection hybrid hit -> chunk_id -> document_collection raw_text/context")
    finally:
        client.close()


if __name__ == "__main__":
    main()
