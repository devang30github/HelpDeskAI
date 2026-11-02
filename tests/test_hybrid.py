import pytest

from app.rag import hybrid  # noqa: F401  (registers the retriever)
from app.rag.hybrid import HybridRetriever, reciprocal_rank_fusion
from app.rag.retrieval import _REGISTRY
from app.rag.types import Chunk, RetrievedChunk


def _chunks():
    return [
        Chunk("a::chunk-000", "a", "a.txt", 0, "APP-318 means the catalogue reports an installation rollback.", {"title": "Software Guide"}),
        Chunk("b::chunk-000", "b", "b.txt", 0, "Restart the application and check for updates in the portal.", {"title": "Crash Guide"}),
        Chunk("c::chunk-000", "c", "c.txt", 0, "Clear the cache and sign in again after a restart.", {"title": "Cache Guide"}),
    ]


class FakeDense:
    name = "fake_dense"

    def __init__(self, order):
        self.order = order

    def retrieve(self, query, k):
        by_id = {c.chunk_id: c for c in _chunks()}
        return [RetrievedChunk(cid, by_id[cid].doc_id, by_id[cid].source_name, by_id[cid].text,
                               0.9 - 0.1 * i, i + 1, by_id[cid].metadata)
                for i, cid in enumerate(self.order[:k])]


def test_rrf_combines_rankings():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]], k=60)
    assert [item for item, _ in fused] == ["a", "c", "b"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_single_list_empty_input_and_deterministic_ties():
    assert [i for i, _ in reciprocal_rank_fusion([["x", "y", "z"]])] == ["x", "y", "z"]
    assert reciprocal_rank_fusion([]) == [] and reciprocal_rank_fusion([[], []]) == []
    assert [i for i, _ in reciprocal_rank_fusion([["x"], ["y"]])] == ["x", "y"]  # tie: earlier list wins


def test_exact_code_match_is_promoted_over_dense_order():
    retriever = HybridRetriever(FakeDense(["b::chunk-000", "c::chunk-000", "a::chunk-000"]), _chunks())
    hits = retriever.retrieve("APP-318", 2)
    assert [h.doc_id for h in hits] == ["a", "b"] and [h.rank for h in hits] == [1, 2]
    assert hits[0].score > hits[1].score
    assert hits[0].metadata["dense_rank"] == 3 and hits[0].metadata["bm25_rank"] == 1
    assert hits[0].metadata["title"] == "Software Guide"


def test_without_sparse_matches_the_dense_order_is_kept():
    retriever = HybridRetriever(FakeDense(["b::chunk-000", "c::chunk-000", "a::chunk-000"]), _chunks())
    assert [h.doc_id for h in retriever.retrieve("qzx", 3)] == ["b", "c", "a"]


def test_empty_query_zero_k_and_params():
    retriever = HybridRetriever(FakeDense(["a::chunk-000"]), _chunks(), rrf_k=60, candidate_k=20)
    assert retriever.retrieve("   ", 3) == [] and retriever.retrieve("APP-318", 0) == []
    assert retriever.params["rrf_k"] == 60 and retriever.params["fusion"] == "rrf"


def test_the_hybrid_retriever_is_registered():
    assert "hybrid_rrf_v2" in _REGISTRY