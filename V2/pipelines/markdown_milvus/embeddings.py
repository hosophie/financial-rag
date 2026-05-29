from __future__ import annotations

import hashlib
import math
from typing import Protocol

from .schema import EMBED_DIM


class EmbeddingProvider(Protocol):
    dim: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_text(self, text: str) -> list[float]:
        ...


class HashEmbeddingProvider:
    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in iter_char_ngrams(text):
            add_feature(vector, f"gram:{token}", 0.25)
        for token in extract_ascii_words(text):
            add_feature(vector, f"word:{token.lower()}", 0.75)
        for marker in ("研发", "现金流", "毛利率", "风险", "销量", "海外", "表格", "图像"):
            if marker in text:
                add_feature(vector, f"marker:{marker}", 2.0)

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def add_feature(vector: list[float], key: str, weight: float) -> None:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % len(vector)
    vector[index] += weight


def iter_char_ngrams(text: str, n: int = 2) -> list[str]:
    compact = "".join(ch for ch in text if not ch.isspace())
    if len(compact) < n:
        return [compact] if compact else []
    return [compact[index : index + n] for index in range(len(compact) - n + 1)]


def extract_ascii_words(text: str) -> list[str]:
    words: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isascii() and char.isalnum():
            current.append(char)
        elif current:
            words.append("".join(current))
            current.clear()
    if current:
        words.append("".join(current))
    return words
