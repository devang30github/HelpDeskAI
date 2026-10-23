"""Query the index from the CLI.   python -m scripts.search "my vpn times out" -k 3"""
import argparse
import sys

from app.core.config import get_settings
from app.rag.retrieval import get_retriever
from app.rag.vector_store import IndexNotFound


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=None)
    args = parser.parse_args()
    k = args.k or get_settings().retrieval_top_k

    try:
        retriever = get_retriever()
    except IndexNotFound:
        print("No index found. Run: python -m scripts.ingest")
        return 1

    for hit in retriever.retrieve(args.query, k):
        print(f"#{hit.rank}  score={hit.score:.3f}  {hit.chunk_id}")
        print("    " + hit.text[:220].replace("\n", " ") + "...")
    return 0


if __name__ == "__main__":
    sys.exit(main())