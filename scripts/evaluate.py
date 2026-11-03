"""Run the evaluation.
   python -m scripts.evaluate --split smoke                  # quick check, uses the LLM
   python -m scripts.evaluate --split dev --retrieval-only   # deterministic, no API calls
   python -m scripts.evaluate --split test --name v1-baseline
Experiment settings must reach the app's configuration BEFORE any app module is imported, so
this script prepares the environment first and imports the rest lazily."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _bootstrap_env(argv: list[str]) -> tuple[dict, str]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(REPO / "configs" / "experiments" / "v1_baseline.json"))
    args, _ = pre.parse_known_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if cfg.get("retriever"):
        os.environ["RETRIEVER_NAME"] = cfg["retriever"]
    if cfg.get("top_k"):
        os.environ["RETRIEVAL_TOP_K"] = str(cfg["top_k"])
    if cfg.get("index_dir"):
        os.environ["INDEX_DIR"] = str(Path(cfg["index_dir"]).resolve())
    # A run must be reproducible from its config alone, so the prompt version never comes from .env.
    os.environ["PROMPT_VERSION"] = cfg.get("prompt_version", "v1")
    hybrid = cfg.get("hybrid") or {}
    for key, env in (("rrf_k", "HYBRID_RRF_K"), ("candidate_k", "HYBRID_CANDIDATE_K"),
                     ("bm25_k1", "BM25_K1"), ("bm25_b", "BM25_B")):
        if key in hybrid:
            os.environ[env] = str(hybrid[key])
    # An evaluation must never touch the real tickets database.
    os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp(prefix='hd-eval-')}/eval.db"
    return cfg, args.config


def main() -> int:
    argv = sys.argv[1:]
    cfg, cfg_path = _bootstrap_env(argv)
    from app.evaluation.cli import run_cli

    return run_cli(argv, cfg, cfg_path)


if __name__ == "__main__":
    sys.exit(main())