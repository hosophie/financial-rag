from __future__ import annotations

from pymilvus import DataType, Function, FunctionType, MilvusClient


DOCUMENT_COLLECTION_NAME = "financial_document_collection_v2"
QUESTION_COLLECTION_NAME = "financial_question_collection_v2"

EMBED_DIM = 1024
DOC_LOOKUP_FIELD = "_lookup_vec"
DENSE_FIELD = "dense_vec"
SPARSE_FIELD = "sparse_vec"
QUESTION_TEXT_FIELD = "question_text"

DOCUMENT_OUTPUT_FIELDS = [
    "chunk_id",
    "doc_id",
    "company",
    "ticker",
    "year",
    "section",
    "chunk_type",
    "seq",
    "raw_text",
    "prev_chunk_id",
    "next_chunk_id",
    "context",
]

QUESTION_OUTPUT_FIELDS = [
    "question_id",
    "chunk_id",
    "question_text",
    "company",
    "ticker",
    "year",
    "section",
    "chunk_type",
]


def create_document_schema():
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="company", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="ticker", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="year", datatype=DataType.INT64)
    schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="chunk_type", datatype=DataType.VARCHAR, max_length=16)
    schema.add_field(field_name="seq", datatype=DataType.INT64)
    schema.add_field(field_name="raw_text", datatype=DataType.VARCHAR, max_length=16384)
    schema.add_field(field_name="prev_chunk_id", datatype=DataType.VARCHAR, max_length=128, nullable=True)
    schema.add_field(field_name="next_chunk_id", datatype=DataType.VARCHAR, max_length=128, nullable=True)
    schema.add_field(field_name="context", datatype=DataType.VARCHAR, max_length=4096, nullable=True)
    schema.add_field(field_name=DOC_LOOKUP_FIELD, datatype=DataType.FLOAT_VECTOR, dim=1)
    return schema


def create_document_index_params():
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name=DOC_LOOKUP_FIELD,
        index_name="document_lookup_vec_index",
        index_type="AUTOINDEX",
        metric_type="L2",
    )
    return index_params


def create_question_schema():
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="question_id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(
        field_name=QUESTION_TEXT_FIELD,
        datatype=DataType.VARCHAR,
        max_length=2048,
        enable_analyzer=True,
        analyzer_params={"tokenizer": "jieba"},
    )
    schema.add_field(field_name=DENSE_FIELD, datatype=DataType.FLOAT_VECTOR, dim=EMBED_DIM)
    schema.add_field(field_name=SPARSE_FIELD, datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="company", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="ticker", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="year", datatype=DataType.INT64)
    schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="chunk_type", datatype=DataType.VARCHAR, max_length=16)
    schema.add_function(
        Function(
            name="question_bm25",
            input_field_names=[QUESTION_TEXT_FIELD],
            output_field_names=[SPARSE_FIELD],
            function_type=FunctionType.BM25,
        )
    )
    return schema


def create_question_index_params():
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name=DENSE_FIELD,
        index_name="question_dense_index",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    index_params.add_index(
        field_name=SPARSE_FIELD,
        index_name="question_sparse_bm25_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
    )
    return index_params
