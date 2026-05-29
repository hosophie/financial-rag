#!/usr/bin/env python
"""Build the V2 Milvus Lite index from raw markdown reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from pipelines.markdown_milvus.embeddings import HashEmbeddingProvider
    from pipelines.markdown_milvus.milvus_store import get_collection_count
    from pipelines.markdown_milvus.pipeline import build_milvus_index
    from pipelines.markdown_milvus.schema import DOCUMENT_COLLECTION_NAME, QUESTION_COLLECTION_NAME
except ImportError as exc:
    print(
        "Milvus V2 dependencies are missing. Install requirements first:\n"
        "  /root/miniconda3/bin/python -m pip install -r V2/requirements.txt\n"
        f"Import error: {exc}"
    )
    sys.exit(1)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_REPORTS_DIR = ROOT_DIR / "data" / "raw_reports"
DEFAULT_DB_PATH = ROOT_DIR / "milvus_lite_index.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_REPORTS_DIR,
        help="Directory containing raw markdown annual reports.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Milvus Lite database path to create or replace.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding_provider = HashEmbeddingProvider()
    client, result = build_milvus_index(
        raw_reports_dir=args.raw_dir,
        db_path=args.db_path,
        embedding_provider=embedding_provider,
    )
    try:
        document_rows = get_collection_count(client, DOCUMENT_COLLECTION_NAME, "chunk_id")
        question_rows = get_collection_count(client, QUESTION_COLLECTION_NAME, "question_id")
        chunk_types = sorted({chunk.chunk_type for chunk in result.chunks})

        print("Milvus index build complete")
        print(f"- Raw markdown dir: {result.raw_reports_dir}")
        print(f"- DB path: {result.db_path}")
        print(f"- Document chunks built: {len(result.chunks)}")
        print(f"- Question records built: {len(result.questions)}")
        print(f"- Milvus document rows: {document_rows}")
        print(f"- Milvus question rows: {question_rows}")
        print(f"- Chunk types: {', '.join(chunk_types)}")
        print("- Embedding provider: HashEmbeddingProvider(dim=1024)")
    finally:
        client.close()


if __name__ == "__main__":
    main()
