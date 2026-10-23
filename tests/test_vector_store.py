import numpy as np
import pytest

from app.rag.types import Chunk
from app.rag.vector_store import (
    CHUNKS_FILE,
    IndexCorrupt,
    IndexNotFound,
    VectorStore,
    index_status,
)


def _make(n=5, dim=8):
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(n, dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    chunks = [
        Chunk(chunk_id=f"d{i}::chunk-000", doc_id=f"d{i}", source_name=f"d{i}.txt",
              chunk_index=0, text=f"text {i}")
        for i in range(n)
    ]
    return chunks, vecs


def test_search_returns_matching_chunk_first():
    chunks, vecs = _make()
    store = VectorStore.build(chunks, vecs, {"embedding_model": "test-model"})
    results = store.search(vecs[2], 3)
    assert results[0][0].chunk_id == "d2::chunk-000"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
    assert len(results) == 3


def test_save_load_roundtrip_keeps_vector_chunk_association(tmp_path):
    chunks, vecs = _make()
    VectorStore.build(chunks, vecs, {"embedding_model": "test-model"}).save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    for i in range(5):
        assert loaded.search(vecs[i], 1)[0][0].chunk_id == f"d{i}::chunk-000"
    assert loaded.manifest["embedding_model"] == "test-model"
    assert loaded.manifest["num_chunks"] == 5


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(IndexNotFound):
        VectorStore.load(tmp_path / "nope")


def test_count_mismatch_is_detected(tmp_path):
    chunks, vecs = _make()
    VectorStore.build(chunks, vecs, {"embedding_model": "test-model"}).save(tmp_path)
    path = tmp_path / CHUNKS_FILE
    path.write_text("\n".join(path.read_text().splitlines()[:3]) + "\n")
    with pytest.raises(IndexCorrupt):
        VectorStore.load(tmp_path)


def test_index_status(tmp_path):
    assert index_status(tmp_path) == {"built": False}
    chunks, vecs = _make()
    VectorStore.build(chunks, vecs, {"embedding_model": "test-model"}).save(tmp_path)
    status = index_status(tmp_path)
    assert status["built"] is True and status["num_chunks"] == 5