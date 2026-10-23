"""Retriever interface + registry.

Version 2 adds retrievers (BM25, hybrid+RRF, reranked) by registering new names;
callers select one via the RETRIEVER_NAME setting / experiment config.
"""
from typing import Callable, Protocol

from app.core.config import Settings, get_settings
from app.rag.embeddings import Embedder, get_embedder
from app.rag.types import RetrievedChunk
from app.rag.vector_store import IndexMismatch, VectorStore


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]: ...


class DenseRetriever:
    """V1 baseline: dense vector similarity over FAISS. Nothing else."""

    name = "dense_v1"

    def __init__(self, store: VectorStore, embedder: Embedder):
        indexed_with = store.manifest.get("embedding_model")
        if indexed_with != embedder.model_name:
            raise IndexMismatch(
                f"Index was built with '{indexed_with}' but '{embedder.model_name}' is configured. "
                "Rebuild with: python -m scripts.ingest"
            )
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        query_vector = self.embedder.encode([query])[0]
        hits = self.store.search(query_vector, k)
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source_name=chunk.source_name,
                text=chunk.text,
                score=score,
                rank=rank,
                metadata=chunk.metadata,
            )
            for rank, (chunk, score) in enumerate(hits, start=1)
        ]


_REGISTRY: dict[str, Callable[[Settings], Retriever]] = {}
_CACHE: dict[str, Retriever] = {}


def register_retriever(name: str):
    def decorator(factory: Callable[[Settings], Retriever]):
        _REGISTRY[name] = factory
        return factory

    return decorator


@register_retriever("dense_v1")
def _build_dense(settings: Settings) -> Retriever:
    store = VectorStore.load(settings.index_dir)
    return DenseRetriever(store, get_embedder(settings.embedding_model))


def get_retriever(name: str | None = None) -> Retriever:
    settings = get_settings()
    name = name or settings.retriever_name
    if name not in _REGISTRY:
        raise KeyError(f"Unknown retriever '{name}'. Available: {sorted(_REGISTRY)}")
    if name not in _CACHE:
        _CACHE[name] = _REGISTRY[name](settings)
    return _CACHE[name]


def reset_retriever_cache() -> None:
    _CACHE.clear()