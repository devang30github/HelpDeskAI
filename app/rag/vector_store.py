"""FAISS index + chunk records + manifest, persisted and reloaded together.

Files in index_dir:
  faiss.index    IndexIDMap(IndexFlatIP) - vector IDs are explicit int64s
  chunks.jsonl   one chunk per line, including its vector_id
  manifest.json  embedding model, dim, chunk params, file hashes, counts
"""
import json
import os
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from app.rag.types import Chunk

INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


class IndexNotFound(Exception):
    pass


class IndexCorrupt(Exception):
    pass


class IndexMismatch(Exception):
    pass


def read_manifest(index_dir: Path) -> dict:
    path = Path(index_dir) / MANIFEST_FILE
    if not path.exists():
        raise IndexNotFound(f"No index manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def index_status(index_dir: Path) -> dict:
    """Cheap status check (reads the manifest only; does not load FAISS)."""
    try:
        m = read_manifest(index_dir)
    except IndexNotFound:
        return {"built": False}
    return {
        "built": True,
        "num_chunks": m.get("num_chunks"),
        "num_documents": m.get("num_documents"),
        "embedding_model": m.get("embedding_model"),
        "built_at": m.get("built_at"),
    }


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


class VectorStore:
    def __init__(self, index, chunks: list[Chunk], manifest: dict):
        self.index = index
        self.chunks = chunks
        self.manifest = manifest
        self._by_vector_id = {c.vector_id: c for c in chunks}

    @classmethod
    def build(cls, chunks: list[Chunk], vectors: np.ndarray, manifest: dict) -> "VectorStore":
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if len(chunks) == 0:
            raise ValueError("cannot build an index with no chunks")
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        dim = vectors.shape[1]
        ids = np.arange(len(chunks), dtype=np.int64)
        for chunk, vid in zip(chunks, ids):
            chunk.vector_id = int(vid)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        index.add_with_ids(vectors, ids)
        manifest = {
            **manifest,
            "embedding_dim": int(dim),
            "num_chunks": len(chunks),
            "index_format": 1,
        }
        return cls(index, chunks, manifest)

    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        index_tmp = index_dir / (INDEX_FILE + ".tmp")
        faiss.write_index(self.index, str(index_tmp))
        os.replace(index_tmp, index_dir / INDEX_FILE)

        lines = [json.dumps(asdict(c), ensure_ascii=False) for c in self.chunks]
        _atomic_write_text(index_dir / CHUNKS_FILE, "\n".join(lines) + "\n")
        # Manifest last: its presence signals a complete index.
        _atomic_write_text(
            index_dir / MANIFEST_FILE, json.dumps(self.manifest, indent=2, ensure_ascii=False)
        )

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        index_dir = Path(index_dir)
        manifest = read_manifest(index_dir)
        index_path = index_dir / INDEX_FILE
        chunks_path = index_dir / CHUNKS_FILE
        if not index_path.exists() or not chunks_path.exists():
            raise IndexNotFound(f"Index files missing in {index_dir}")

        index = faiss.read_index(str(index_path))
        chunks = [
            Chunk(**json.loads(line))
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if index.ntotal != len(chunks):
            raise IndexCorrupt(
                f"FAISS holds {index.ntotal} vectors but chunks.jsonl has {len(chunks)} records. "
                "Rebuild with: python -m scripts.ingest"
            )
        return cls(index, chunks, manifest)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        k = min(k, self.index.ntotal)
        if k <= 0:
            return []
        q = np.ascontiguousarray(query_vector, dtype="float32").reshape(1, -1)
        scores, ids = self.index.search(q, k)
        results = []
        for vid, score in zip(ids[0], scores[0]):
            if vid == -1:
                continue
            results.append((self._by_vector_id[int(vid)], float(score)))
        return results