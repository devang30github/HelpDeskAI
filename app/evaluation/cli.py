"""Command-line orchestration for evaluation runs (imported after the environment is prepared)."""
import argparse
import importlib.metadata as md
import logging
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import BASE_DIR, get_settings
from app.evaluation import dataset as ds
from app.evaluation.runner import run_evaluation
from app.rag.generation import LLMConfigError
from app.rag.prompts import PROMPT_VERSION
from app.rag.vector_store import IndexNotFound, read_manifest

DEFAULT_DATASET = BASE_DIR / "data" / "evaluation" / "qa_v1.jsonl"


def _git_info() -> dict:
    try:
        def run(*cmd):
            return subprocess.run(cmd, capture_output=True, text=True, timeout=5, cwd=BASE_DIR).stdout.strip()
        return {"commit": run("git", "rev-parse", "HEAD") or None, "dirty": bool(run("git", "status", "--porcelain"))}
    except Exception:
        return {"commit": None, "dirty": None}


def _versions(names: list[str]) -> dict:
    out = {}
    for name in names:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def _resolve_run_dir(arg: str, out_dir: str) -> Path | None:
    for candidate in (Path(arg), Path(out_dir) / arg, BASE_DIR / "reports" / "baselines" / arg):
        if (candidate / "config.json").exists():
            return candidate
    return None


def _run_extras(run_dir: Path, *, ragas: bool, tickets: bool, classification: bool, allow_unvalidated: bool) -> int:
    from app.evaluation import extras

    stages = []
    if ragas:
        stages.append(("RAGAS", lambda: extras.apply_ragas(run_dir, progress=print)))
    if tickets:
        stages.append(("Ticket flow", lambda: extras.apply_tickets(run_dir, allow_unvalidated, progress=print)))
    if classification:
        stages.append(("Classification", lambda: extras.apply_classification(run_dir, allow_unvalidated, progress=print)))

    status = 0
    for name, stage in stages:
        print(f"\n{name} evaluation...")
        try:
            stage()
        except Exception as e:
            print(f"{name} evaluation failed: {type(e).__name__}: {e}")
            if name == "RAGAS":
                print("Run `python -m scripts.check_ragas` to diagnose the RAGAS wiring.")
            status = 1
    return status


def run_cli(argv: list[str], cfg: dict, cfg_path: str) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "sentence_transformers", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Run the HelpDeskAI evaluation")
    parser.add_argument("--config", default=cfg_path)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--split", choices=["smoke", "dev", "test", "full"], default="dev")
    parser.add_argument("--name", default=cfg.get("name", "run"))
    parser.add_argument("--limit", type=int, help="only the first N selected questions")
    parser.add_argument("--retrieval-only", action="store_true", help="deterministic retrieval metrics, no LLM calls")
    parser.add_argument("--ragas", action="store_true", help="score the generations with RAGAS (LLM judge)")
    parser.add_argument("--tickets", action="store_true", help="evaluate the ticket flow (uses the LLM)")
    parser.add_argument("--classification", action="store_true", help="evaluate category classification (uses the LLM)")
    parser.add_argument("--all", action="store_true", help="shorthand for --ragas --tickets --classification")
    parser.add_argument("--rescore", metavar="RUN", help="run the selected add-on stages on an existing run (default: RAGAS)")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds between LLM calls (rate limits)")
    parser.add_argument("--allow-unvalidated", action="store_true")
    parser.add_argument("--allow-kb-mismatch", action="store_true")
    parser.add_argument("--out-dir", default=str(BASE_DIR / "reports" / "runs"))
    args = parser.parse_args(argv)

    want_ragas, want_tickets, want_class = (args.ragas or args.all, args.tickets or args.all,
                                            args.classification or args.all)

    if args.rescore:
        run_dir = _resolve_run_dir(args.rescore, args.out_dir)
        if run_dir is None:
            print(f"Could not find a run folder for '{args.rescore}'")
            return 1
        if not (want_ragas or want_tickets or want_class):
            want_ragas = True
        status = _run_extras(run_dir, ragas=want_ragas, tickets=want_tickets, classification=want_class,
                             allow_unvalidated=args.allow_unvalidated)
        print(f"\nReport: {run_dir / 'report.md'}")
        return status

    if args.retrieval_only and want_ragas:
        print("--ragas needs generations; drop --retrieval-only.")
        return 1

    settings = get_settings()
    try:
        all_items = ds.load_items(args.dataset)
        items = ds.select_items(all_items, args.split)
    except FileNotFoundError:
        print(f"Dataset not found: {args.dataset}")
        return 1
    except ds.DatasetError as e:
        print("Dataset problem:\n" + str(e))
        return 1
    if args.limit:
        items = items[: args.limit]
    meta = ds.load_meta(args.dataset)

    unvalidated = [i.id for i in items if not i.validated]
    if unvalidated:
        if args.split in ("test", "full") and not args.allow_unvalidated:
            print(f"Refusing to run split '{args.split}' with {len(unvalidated)} unvalidated questions "
                  f"(e.g. {unvalidated[:5]}). Validate them, or pass --allow-unvalidated.")
            return 1
        print(f"warning: {len(unvalidated)} of {len(items)} selected questions are not hand-validated yet")

    try:
        manifest = read_manifest(settings.index_dir)
    except IndexNotFound:
        print("No index found. Run: python -m scripts.ingest")
        return 1
    kb_problems = ds.check_kb_pin(meta, manifest.get("files", {}))
    if kb_problems and not args.allow_kb_mismatch:
        print("The indexed knowledge base does not match the dataset's KB pin:")
        for p in kb_problems:
            print(f"  {p}")
        print("Rebuild the index, re-pin with 'python -m scripts.validate_dataset --pin-kb' if the KB change is "
              "intentional (then re-validate the labels), or pass --allow-kb-mismatch.")
        return 1

    from app.rag.pipeline import get_pipeline
    from app.rag.retrieval import get_retriever

    try:
        retriever = get_retriever()
        pipeline = None if args.retrieval_only else get_pipeline()
    except LLMConfigError as e:
        print(f"Configuration problem: {e}")
        return 1

    eval_k = max(int(cfg.get("retrieval_eval_k", 10)), settings.retrieval_top_k)
    k_values = sorted({k for k in (1, 3, 5, settings.retrieval_top_k, eval_k) if k <= eval_k})
    slug = re.sub(r"[^a-z0-9]+", "-", args.name.lower()).strip("-") or "run"
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{slug}"
    out_dir = Path(args.out_dir) / run_id

    run_config = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": args.name,
        "split": args.split,
        "retrieval_only": args.retrieval_only,
        "limit": args.limit,
        "experiment_config_path": args.config,
        "experiment_config": cfg,
        "dataset": {
            "path": str(args.dataset),
            "version": meta.get("version"),
            "file_sha256": ds.file_sha256(args.dataset),
            "label_hash": ds.label_hash(items),
            "items": len(items),
        },
        "kb": {"pin_ok": not kb_problems, "pin_problems": kb_problems, "indexed_files": manifest.get("files")},
        "index": {k: manifest.get(k) for k in ("embedding_model", "chunk_size", "chunk_overlap", "built_at")},
        "retrieval": {"retriever": settings.retriever_name, "top_k": settings.retrieval_top_k, "eval_k": eval_k},
        "generation": pipeline.config() if pipeline else None,
        "prompt_version": PROMPT_VERSION,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git": _git_info(),
            "packages": _versions(["faiss-cpu", "sentence-transformers", "openai", "ragas", "instructor",
                                   "langchain-community", "numpy"]),
        },
    }

    print(f"Run {run_id}: {len(items)} questions, split '{args.split}'"
          f"{' (retrieval only)' if args.retrieval_only else ''}")
    summary = run_evaluation(
        items=items, retriever=retriever, pipeline=pipeline, out_dir=out_dir, run_config=run_config,
        k_values=k_values, eval_k=eval_k, headline_k=settings.retrieval_top_k,
        pricing=cfg.get("pricing"), sleep=args.sleep, progress=print,
    )

    overall, hk = summary["retrieval"]["overall"], settings.retrieval_top_k
    print(f"\nRetrieval (N={summary['retrieval']['n']}): Hit@{hk} {overall.get(f'hit@{hk}', float('nan')):.3f}  "
          f"MRR@{hk} {overall.get(f'mrr@{hk}', float('nan')):.3f}")
    if "abstention" in summary:
        ab = summary["abstention"]
        print(f"Abstention recall {ab['abstention_recall']}  false abstention rate {ab['false_abstention_rate']}")
        print(f"Generation: {summary['generation']['n_ok']} ok, {summary['generation']['n_failed']} failed")

    status = 0
    if want_ragas or want_tickets or want_class:
        status = _run_extras(out_dir, ragas=want_ragas, tickets=want_tickets, classification=want_class,
                             allow_unvalidated=args.allow_unvalidated)
    print(f"\nReport:  {out_dir / 'report.md'}\nArtifacts in {out_dir}")
    return status