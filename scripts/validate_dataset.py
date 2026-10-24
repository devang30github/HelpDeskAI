"""Validate the evaluation dataset against the knowledge base.
   python -m scripts.validate_dataset                    # schema + KB checks + coverage
   python -m scripts.validate_dataset --show q001        # one row with the referenced documents
   python -m scripts.validate_dataset --probe-retrieval  # closest documents for unanswerable rows
   python -m scripts.validate_dataset --pin-kb           # record the current KB file hashes in dataset_meta.json"""
import argparse
import json
import sys

from app.core.config import BASE_DIR, get_settings
from app.evaluation import dataset as ds
from app.rag.ingestion import fingerprint_kb, load_documents

DEFAULT_DATASET = BASE_DIR / "data" / "evaluation" / "qa_v1.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--show", metavar="ID")
    parser.add_argument("--probe-retrieval", action="store_true")
    parser.add_argument("--pin-kb", action="store_true")
    args = parser.parse_args()
    settings = get_settings()

    try:
        items = ds.load_items(args.dataset)
    except FileNotFoundError:
        print(f"Dataset not found: {args.dataset}")
        return 1
    except ds.DatasetError as e:
        print("Dataset errors:\n" + str(e))
        return 1

    if args.show:
        item = next((i for i in items if i.id == args.show), None)
        if item is None:
            print(f"No question with id '{args.show}'")
            return 1
        print(json.dumps(item.model_dump(), indent=2, ensure_ascii=False))
        docs, _ = load_documents(settings.kb_dir)
        by_id = {d.doc_id: d for d in docs}
        for doc_id in item.reference_doc_ids:
            print(f"\n===== {doc_id} =====")
            print(by_id[doc_id].text if doc_id in by_id else "(document not found)")
        return 0

    problems = ds.validate_against_kb(items, settings.kb_dir)
    cov = ds.coverage(items)
    print(f"{cov['total']} questions: {cov['answerable']} answerable, {cov['unanswerable']} unanswerable, "
          f"{cov['validated']} validated, {cov['smoke']} smoke")
    print(f"  splits: {cov['by_split']}   origins: {cov['origins']}")
    print(f"  types: {cov['by_type']}")
    print(f"  answerable by category: {cov['answerable_by_category']}")
    for w in ds.coverage_warnings(items):
        print(f"  warning: {w}")
    for p in problems:
        print(f"  ERROR: {p}")

    if args.pin_kb:
        meta = ds.load_meta(args.dataset) or {"dataset_name": "helpdesk-qa", "version": "1.0"}
        meta["kb_files"] = fingerprint_kb(settings.kb_dir)
        ds.meta_path(args.dataset).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nPinned {len(meta['kb_files'])} knowledge-base files in {ds.meta_path(args.dataset).name}")

    meta = ds.load_meta(args.dataset)
    pin_problems = ds.check_kb_pin(meta, fingerprint_kb(settings.kb_dir))
    print("\nKB pin: " + ("matches the current knowledge base" if not pin_problems else "MISMATCH"))
    for p in pin_problems:
        print(f"  {p}")

    if args.probe_retrieval:
        from app.rag.retrieval import get_retriever

        retriever = get_retriever()
        print("\nClosest documents for unanswerable questions (check that none truly answers them):")
        for item in (i for i in items if not i.answerable):
            hits = retriever.retrieve(item.question, 3)
            top = ", ".join(f"{h.doc_id} ({h.score:.2f})" for h in hits)
            print(f"  {item.id}: {item.question}\n      -> {top}")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())