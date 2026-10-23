"""Measure retrieval latency.   python -m scripts.bench_retrieval -n 30
Reports the cold first call, then median/p95 for embedding, FAISS search and the total."""
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

    t = time.perf_counter()
    retriever.retrieve(QUERIES[0], k)
    print(f"cold first call: {ms(t):.0f} ms (model load + first search)")

    embed, search, total = [], [], []
    for i in range(args.n):
        q = QUERIES[i % len(QUERIES)]
        t = time.perf_counter()
        vec = retriever.embedder.encode([q])[0]
        e = ms(t)
        t = time.perf_counter()
        retriever.store.search(vec, k)
        s = ms(t)
        embed.append(e)
        search.append(s)
        t = time.perf_counter()
        retriever.retrieve(q, k)
        total.append(ms(t))

    print(f"\n{args.n} warm queries (top_k={k}):")
    for name, vals in (("embed", embed), ("faiss search", search), ("retrieve total", total)):
        print(f"  {name:<15} median {statistics.median(vals):6.1f} ms   p95 {pct(vals, 95):6.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())