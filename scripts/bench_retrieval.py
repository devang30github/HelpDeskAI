"""Measure retrieval latency.   python -m scripts.bench_retrieval -n 30
   RETRIEVER_NAME=hybrid_rrf_v2 python -m scripts.bench_retrieval -n 30
Reports the cold first call, then median/p95 for each stage. For the hybrid retriever the dense
part, the BM25 part and the full retrieve() are timed separately."""
import argparse
import statistics
import sys
import time

from app.core.config import get_settings
from app.rag.retrieval import get_retriever
from app.rag.vector_store import IndexNotFound

QUERIES = [
    "my vpn keeps timing out",
    "laptop battery is swollen",
    "i am locked out of my account",
    "printer shows offline",
    "how do i request access to a shared folder",
    "meeting room projector is not working",
]


def pct(values: list[float], p: int) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))]


def ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=30, help="number of timed queries")
    args = parser.parse_args()
    k = get_settings().retrieval_top_k

    try:
        retriever = get_retriever()
    except IndexNotFound:
        print("No index found. Run: python -m scripts.ingest")
        return 1

    name = getattr(retriever, "name", type(retriever).__name__)
    dense = getattr(retriever, "dense", retriever)  # the hybrid retriever wraps a dense one
    bm25 = getattr(retriever, "bm25", None)
    pool = max(getattr(retriever, "candidate_k", k), k)
    print(f"retriever: {name} | top_k={k}" + (f" | candidate pool {pool}" if bm25 else ""))

    t = time.perf_counter()
    retriever.retrieve(QUERIES[0], k)
    print(f"cold first call: {ms(t):.0f} ms (model load + first search)")

    stages: dict[str, list[float]] = {"embed": [], "faiss search": []}
    if bm25:
        stages["bm25 search"] = []
    stages["retrieve total"] = []

    for i in range(args.n):
        q = QUERIES[i % len(QUERIES)]
        t = time.perf_counter()
        vec = dense.embedder.encode([q])[0]
        stages["embed"].append(ms(t))
        t = time.perf_counter()
        dense.store.search(vec, k)
        stages["faiss search"].append(ms(t))
        if bm25:
            t = time.perf_counter()
            bm25.search(q, pool)
            stages["bm25 search"].append(ms(t))
        t = time.perf_counter()
        retriever.retrieve(q, k)
        stages["retrieve total"].append(ms(t))

    print(f"\n{args.n} warm queries:")
    for stage, vals in stages.items():
        print(f"  {stage:<15} median {statistics.median(vals):6.1f} ms   p95 {pct(vals, 95):6.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())