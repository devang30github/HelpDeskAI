"""Okapi BM25 over the knowledge-base chunks (pure Python, deterministic).

The tokenizer keeps hyphenated tokens such as error codes whole and ALSO emits their parts
and the joined form, so "VPN-612" is searchable as "vpn-612", "vpn" and "612", and "Wi-Fi" as
"wi-fi", "wi", "fi" and "wifi". Chunk titles are indexed too (sparse side only)."""
import math
import re
from collections import Counter, defaultdict

from app.rag.types import Chunk

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
_STOPWORDS = frozenset(
    """a an the and or but if then of to in on at by for with from as is are was were be been it its
    this that these those i me my we our you your he she they them do does did done can could should
    would will what when where which who whom how why not no yes so than too very just also into
    about after before""".split()
)


def _stem(token: str) -> str:
    """Very light stemming (plurals, -ing, -ed). Applied identically to documents and queries."""
    if len(token) <= 4 or "-" in token or "_" in token or any(ch.isdigit() for ch in token):
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) - 3 >= 4:
        return token[:-3]
    if token.endswith("ed") and len(token) - 2 >= 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN.findall(text.lower()):
        parts = re.split(r"[-_]", raw)
        variants = [raw] if len(parts) == 1 else [raw, *parts, "".join(parts)]
        for variant in variants:
            if variant in _STOPWORDS or (len(variant) < 2 and not variant.isdigit()):
                continue
            tokens.append(_stem(variant))
    return tokens


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._chunks = list(chunks)
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, chunk in enumerate(self._chunks):
            tokens = tokenize(f"{chunk.metadata.get('title', '')} {chunk.text}")
            self._lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self._postings[term].append((index, tf))
        n = len(self._chunks)
        self._avg_length = sum(self._lengths) / n if n else 0.0
        self._idf = {
            term: math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        terms = set(tokenize(query))
        if not terms or not self._chunks or k <= 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in sorted(terms):  # sorted: deterministic floating-point summation order
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for index, tf in postings:
                length_ratio = self._lengths[index] / self._avg_length if self._avg_length else 1.0
                norm = 1 - self.b + self.b * length_ratio
                scores[index] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self._chunks[i], score) for i, score in ranked if score > 0]