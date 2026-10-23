"""Paragraph-aware chunking. Sizes are in characters.

Paragraphs are packed into chunks up to `chunk_size`. Oversized paragraphs are
split by sentence (hard-split as a last resort). Each new chunk starts with up
to `overlap` characters of the previous chunk, trimmed to a word boundary, so a
chunk may exceed chunk_size by at most about `overlap` characters.
"""
import re

from app.rag.types import Chunk, Document

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_long(paragraph: str, size: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        while len(sentence) > size:  # single sentence longer than a chunk
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:size])
            sentence = sentence[size:]
        if not sentence:
            continue
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= size:
            current += " " + sentence
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for p in paragraphs:
        units.extend([p] if len(p) <= chunk_size else _split_long(p, chunk_size))

    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
        elif len(current) + 2 + len(unit) <= chunk_size:
            current += "\n\n" + unit
        else:
            chunks.append(current)
            tail = ""
            if overlap > 0:
                cut = current[-overlap:]
                space = cut.find(" ")
                tail = cut[space + 1:] if space != -1 else cut
            current = f"{tail}\n\n{unit}" if tail else unit
    if current:
        chunks.append(current)
    return chunks


def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}::chunk-{i:03d}",
            doc_id=doc.doc_id,
            source_name=doc.source_name,
            chunk_index=i,
            text=text,
            metadata=dict(doc.metadata),
        )
        for i, text in enumerate(chunk_text(doc.text, chunk_size, overlap))
    ]