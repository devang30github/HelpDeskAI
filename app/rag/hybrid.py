"""Hybrid retrieval: dense (FAISS) + sparse (BM25), merged with reciprocal rank fusion.

RRF score(chunk) = sum over rankers of 1 / (k + rank). It needs no score normalisation, which
matters because cosine similarity and BM25 scores are on different scales. k=60 is the
conventional constant and is fixed, not tuned."""
from app.core.config import Settings
from app.rag.bm25 import BM25Index
from app.rag.embeddings import get_embedder
from app.rag.retrieval import DenseRetriever, register_retriever
from app.rag.types import Chunk, RetrievedChunk
from app.rag.vector_store import VectorStore


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked lists of ids. Ties are broken by best single rank, then by which list the
    id appeared in first (earlier lists win), then by id, so the result is deterministic."""
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    first_list: dict[str, int] = {}
    for list_index, ranking in enumerate(rankings):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            if item not in best_rank or rank < best_rank[item]:
                best_rank[item] = rank
            first_list.setdefault(item, list_index)
    return sorted(
        scores.items(),
        key=lambda kv: (-round(kv[1], 12), best_rank[kv[0]], first_list[kv[0]], kv[0]),
    )


class HybridRetriever:
    name = "hybrid_rrf_v2"

    def __init__(self, dense, chunks: list[Chunk], rrf_k: int = 60, candidate_k: int = 20,
                 bm25_k1: float = 1.5, bm25_b: float = 0.75):
        self.dense = dense  # anything with retrieve(query, k) -> list[RetrievedChunk]
        self._by_id = {c.chunk_id: c for c in chunks}
        self.bm25 = BM25Index(chunks, bm25_k1, bm25_b)
        self.rrf_k, self.candidate_k = rrf_k, candidate_k
        self.params = {"fusion": "rrf", "rrf_k": rrf_k, "candidate_k": candidate_k,
                       "bm25_k1": bm25_k1, "bm25_b": bm25_b, "bm25_includes_title": True}

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if not query.strip() or k <= 0:
            return []
        pool = max(self.candidate_k, k)
        dense_hits = self.dense.retrieve(query, pool)
        sparse_hits = self.bm25.search(query, pool)
        dense_ids = [h.chunk_id for h in dense_hits]
        sparse_ids = [chunk.chunk_id for chunk, _ in sparse_hits]
        dense_rank = {cid: i for i, cid in enumerate(dense_ids, start=1)}
        sparse_rank = {cid: i for i, cid in enumerate(sparse_ids, start=1)}

        results = []
        for rank, (chunk_id, score) in enumerate(reciprocal_rank_fusion([dense_ids, sparse_ids], self.rrf_k)[:k],
                                                 start=1):
            chunk = self._by_id[chunk_id]
            results.append(RetrievedChunk(
                chunk_id=chunk_id, doc_id=chunk.doc_id, source_name=chunk.source_name, text=chunk.text,
                score=score, rank=rank,  # the RRF score, not a cosine similarity
                metadata={**chunk.metadata, "dense_rank": dense_rank.get(chunk_id),
                          "bm25_rank": sparse_rank.get(chunk_id)},
            ))
        return results


@register_retriever("hybrid_rrf_v2")
def _build_hybrid(settings: Settings) -> HybridRetriever:
    store = VectorStore.load(settings.index_dir)
    dense = DenseRetriever(store, get_embedder(settings.embedding_model))
    return HybridRetriever(dense, store.chunks, settings.hybrid_rrf_k, settings.hybrid_candidate_k,
                           settings.bm25_k1, settings.bm25_b)