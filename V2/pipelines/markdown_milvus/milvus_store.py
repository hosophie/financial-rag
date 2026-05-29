from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from .models import DocumentChunk, QuestionRecord
from .schema import (
    DENSE_FIELD,
    DOCUMENT_COLLECTION_NAME,
    DOCUMENT_OUTPUT_FIELDS,
    DOC_LOOKUP_FIELD,
    QUESTION_COLLECTION_NAME,
    QUESTION_OUTPUT_FIELDS,
    SPARSE_FIELD,
    create_document_index_params,
    create_document_schema,
    create_question_index_params,
    create_question_schema,
)


def reset_database(db_path: Path) -> MilvusClient:
    remove_existing_db(db_path)
    client = MilvusClient(uri=str(db_path))
    client.create_collection(
        collection_name=DOCUMENT_COLLECTION_NAME,
        schema=create_document_schema(),
        index_params=create_document_index_params(),
    )
    client.create_collection(
        collection_name=QUESTION_COLLECTION_NAME,
        schema=create_question_schema(),
        index_params=create_question_index_params(),
    )
    return client


def remove_existing_db(db_path: Path) -> None:
    for path in db_path.parent.glob(f"{db_path.name}*"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def insert_documents(client: MilvusClient, chunks: list[DocumentChunk]) -> None:
    if not chunks:
        return
    client.insert(
        collection_name=DOCUMENT_COLLECTION_NAME,
        data=[document_to_entity(chunk) for chunk in chunks],
    )
    client.flush(collection_name=DOCUMENT_COLLECTION_NAME)
    client.load_collection(collection_name=DOCUMENT_COLLECTION_NAME)


def insert_questions(client: MilvusClient, records: list[QuestionRecord]) -> None:
    if not records:
        return
    client.insert(
        collection_name=QUESTION_COLLECTION_NAME,
        data=[question_to_entity(record) for record in records],
    )
    client.flush(collection_name=QUESTION_COLLECTION_NAME)
    client.load_collection(collection_name=QUESTION_COLLECTION_NAME)


def document_to_entity(chunk: DocumentChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "company": chunk.company,
        "ticker": chunk.ticker,
        "year": chunk.year,
        "section": chunk.section,
        "chunk_type": chunk.chunk_type,
        "seq": chunk.seq,
        "raw_text": chunk.raw_text,
        "prev_chunk_id": chunk.prev_chunk_id,
        "next_chunk_id": chunk.next_chunk_id,
        "context": chunk.context,
        DOC_LOOKUP_FIELD: [0.0],
    }


def question_to_entity(record: QuestionRecord) -> dict[str, Any]:
    return {
        "question_id": record.question_id,
        "chunk_id": record.chunk_id,
        "question_text": record.question_text,
        DENSE_FIELD: record.dense_vec,
        "company": record.company,
        "ticker": record.ticker,
        "year": record.year,
        "section": record.section,
        "chunk_type": record.chunk_type,
    }


def get_collection_count(client: MilvusClient, collection_name: str, id_field: str) -> int:
    try:
        stats = client.get_collection_stats(collection_name=collection_name)
        return int(stats.get("row_count", 0))
    except Exception:
        rows = client.query(
            collection_name=collection_name,
            filter=f'{id_field} != ""',
            output_fields=[id_field],
            limit=10000,
        )
        return len(rows)


def hybrid_search_questions(
    client: MilvusClient,
    query_text: str,
    query_vector: list[float],
    limit: int = 5,
    filter_expr: str | None = None,
) -> list[Any]:
    requests = [
        AnnSearchRequest(
            data=[query_vector],
            anns_field=DENSE_FIELD,
            param={"metric_type": "COSINE", "params": {}},
            limit=limit,
            filter=filter_expr,
        ),
        AnnSearchRequest(
            data=[query_text],
            anns_field=SPARSE_FIELD,
            param={},
            limit=limit,
            filter=filter_expr,
        ),
    ]
    result = client.hybrid_search(
        collection_name=QUESTION_COLLECTION_NAME,
        reqs=requests,
        ranker=RRFRanker(k=60),
        limit=limit,
        output_fields=QUESTION_OUTPUT_FIELDS,
    )
    return list(result[0]) if result else []


def fetch_documents_by_chunk_ids(
    client: MilvusClient,
    chunk_ids: list[str],
) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for chunk_id in chunk_ids:
        rows = client.query(
            collection_name=DOCUMENT_COLLECTION_NAME,
            filter=f'chunk_id == "{escape_filter_string(chunk_id)}"',
            output_fields=DOCUMENT_OUTPUT_FIELDS,
            limit=1,
        )
        if rows:
            row = dict(rows[0])
            documents[row["chunk_id"]] = row
    return documents


def question_hits_to_rows(hits: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hit in hits:
        entity = hit_entity(hit)
        score = hit_value(hit, "distance")
        if score is None:
            score = hit_value(hit, "score")
        rows.append(
            {
                "question_id": entity.get("question_id") or hit_value(hit, "id"),
                "chunk_id": entity.get("chunk_id", ""),
                "question_text": entity.get("question_text", ""),
                "company": entity.get("company", ""),
                "ticker": entity.get("ticker", ""),
                "year": entity.get("year", 0),
                "section": entity.get("section", ""),
                "chunk_type": entity.get("chunk_type", ""),
                "score": float(score) if score is not None else 0.0,
            }
        )
    return rows


def hit_value(hit: Any, key: str, default: Any = None) -> Any:
    if isinstance(hit, dict):
        return hit.get(key, default)
    if hasattr(hit, key):
        return getattr(hit, key)
    if hasattr(hit, "get"):
        try:
            return hit.get(key, default)
        except TypeError:
            return hit.get(key)
    return default


def hit_entity(hit: Any) -> dict[str, Any]:
    entity = hit_value(hit, "entity", {})
    if isinstance(entity, dict):
        return entity
    if hasattr(entity, "to_dict"):
        return entity.to_dict()
    return dict(entity) if entity else {}


def escape_filter_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
