import pytest

from app.rag.chunking import chunk_document, chunk_text
from app.rag.types import Document


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", 100, 10) == ["Hello world."]


def test_blank_text_gives_no_chunks():
    assert chunk_text("  \n \n ", 100, 10) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", size, overlap)


def test_chunks_respect_size_and_keep_all_content():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 30 for i in range(10))
    chunks = chunk_text(text, 300, 50)
    assert len(chunks) > 1
    assert all(0 < len(c) <= 300 + 50 + 2 for c in chunks)
    joined = "\n".join(chunks)
    for i in range(10):
        assert f"Paragraph {i}." in joined


def test_overlap_carries_text_into_next_chunk():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 30 for i in range(4))
    chunks = chunk_text(text, 300, 50)
    assert chunks[1].startswith("word")  # tail of chunk 0 repeated at the start
    assert "Paragraph 1." in chunks[1]


def test_oversized_paragraph_is_split():
    paragraph = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_text(paragraph, 200, 0)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_ids_are_stable():
    doc = Document(doc_id="guide", source_name="guide.txt", text="A.\n\nB.", sha256="x",
                   metadata={"title": "Guide"})
    first = [c.chunk_id for c in chunk_document(doc, 100, 10)]
    second = [c.chunk_id for c in chunk_document(doc, 100, 10)]
    assert first == second == ["guide::chunk-000"]