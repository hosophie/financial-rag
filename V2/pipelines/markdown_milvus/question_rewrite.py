from __future__ import annotations

from .models import DocumentChunk
from .summarization import section_label


def rewrite_question(question: str, chunk: DocumentChunk) -> str:
    normalized = question.strip().rstrip("？?") + "？"
    type_hint = {
        "text": "文本内容",
        "table": "表格内容",
        "figure": "图像内容",
    }[chunk.chunk_type]
    prefix = f"关于{chunk.company}{chunk.year}年{section_label(chunk.section)}的{type_hint}，"
    if chunk.company in normalized and str(chunk.year) in normalized:
        return normalized
    return f"{prefix}{normalized}"


def rewrite_questions(questions: list[str], chunk: DocumentChunk) -> list[str]:
    rewritten: list[str] = []
    seen: set[str] = set()
    for question in questions:
        value = rewrite_question(question, chunk)
        if value not in seen:
            seen.add(value)
            rewritten.append(value)
    return rewritten
