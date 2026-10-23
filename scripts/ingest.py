"""Build (or check) the FAISS index.   Run from repo root:  python -m scripts.ingest"""
import argparse
import logging
import sys

from app.core.config import get_settings
from app.rag.ingestion import build_index, check_index


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build the HelpDeskAI knowledge-base index")
    parser.add_argument("--check", action="store_true", help="only report whether the index is stale")
    parser.add_argument("--chunk-size", type=int, help="override CHUNK_SIZE")
    parser.add_argument("--chunk-overlap", type=int, help="override CHUNK_OVERLAP")
    args = parser.parse_args()
    settings = get_settings()

    if args.check:
        stale, reasons = check_index(settings)
        print("STALE" if stale else "UP TO DATE")
        for r in reasons:
            print(f"  - {r}")
        return 1 if stale else 0

    manifest = build_index(settings, args.chunk_size, args.chunk_overlap)
    print(f"\nIndex built at {settings.index_dir}")
    print(f"  model:     {manifest['embedding_model']} (dim {manifest['embedding_dim']})")
    print(f"  chunking:  size={manifest['chunk_size']} overlap={manifest['chunk_overlap']}")
    print(f"  documents: {manifest['num_documents']}   chunks: {manifest['num_chunks']}")
    for doc_id, info in manifest["documents"].items():
        print(f"    {doc_id:<42} {info['num_chunks']:>2} chunks  [{info.get('category')}]")
    for item in manifest["skipped"]:
        print(f"  skipped: {item['file']} ({item['reason']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())