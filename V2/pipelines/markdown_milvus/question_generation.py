from __future__ import annotations

import re
from uuid import uuid4

from .embeddings import EmbeddingProvider
from .models import DocumentChunk, QuestionRecord
from .question_rewrite import rewrite_questions
from .summarization import section_label


KEYWORD_CANDIDATES = [
    "研发",
    "动力电池",
    "储能",
    "现金流",
    "营业收入",
    "毛利率",
    "风险",
    "海外",
    "销量",
    "供应链",
    "价格竞争",
]


def generate_questions(chunk: DocumentChunk) -> list[str]:
    focus = extract_focus(chunk.raw_text)
    label = section_label(chunk.section)
    if chunk.chunk_type == "table":
        candidates = [
            f"{label}表格披露了哪些核心指标？",
            f"{focus}相关表格说明了什么？",
            "经营表现表格里有哪些变化？",
            "表格中的指标变化原因是什么？",
        ]
    elif chunk.chunk_type == "figure":
        candidates = [
            f"{label}图展示了哪些信息？",
            f"{focus}相关图像表达了什么风险或趋势？",
            "这张图可以作为哪些问题的依据？",
        ]
    else:
        candidates = [
            f"{label}部分说明了哪些重点？",
            f"{focus}相关内容有哪些进展？",
            "这段内容对经营表现有什么影响？",
            "管理层对这部分如何解释？",
        ]
    return unique_keep_order(candidates)[:5]


def build_question_records(
    chunks: list[DocumentChunk],
    embedding_provider: EmbeddingProvider,
) -> list[QuestionRecord]:
    question_texts: list[str] = []
    owners: list[DocumentChunk] = []

    for chunk in chunks:
        rewritten = rewrite_questions(generate_questions(chunk), chunk)
        for question in rewritten:
            question_texts.append(question)
            owners.append(chunk)

    vectors = embedding_provider.embed_texts(question_texts)
    records: list[QuestionRecord] = []
    for question_text, chunk, dense_vec in zip(question_texts, owners, vectors):
        records.append(
            QuestionRecord(
                question_id=str(uuid4()),
                chunk_id=chunk.chunk_id,
                question_text=question_text,
                dense_vec=dense_vec,
                company=chunk.company,
                ticker=chunk.ticker,
                year=chunk.year,
                section=chunk.section,
                chunk_type=chunk.chunk_type,
            )
        )
    return records


def extract_focus(text: str) -> str:
    for keyword in KEYWORD_CANDIDATES:
        if keyword in text:
            return keyword
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
    return chinese_terms[0] if chinese_terms else "该内容"


def unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
