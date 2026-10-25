"""Add-on evaluation stages that update an existing run directory:
RAGAS scoring, ticket-flow evaluation and category classification."""
import csv
import json
from pathlib import Path

from app.evaluation import dataset as ds
from app.evaluation import ragas_runner, ticket_eval
from app.evaluation.ragas_setup import METRIC_NAMES
from app.evaluation.report import write_report


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
                    encoding="utf-8")


def load_run(run_dir: Path) -> tuple[dict, dict]:
    return (json.loads((run_dir / "config.json").read_text(encoding="utf-8")),
            json.loads((run_dir / "summary.json").read_text(encoding="utf-8")))


def save_run(run_dir: Path, config: dict, summary: dict) -> None:
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, config, run_dir / "report.md")


def _merge_csv(run_dir: Path, score_rows: list[dict]) -> None:
    path = run_dir / "per_question.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows, fields = list(reader), list(reader.fieldnames or [])
    by_id = {r["id"]: r["scores"] for r in score_rows}
    fields += [n for n in METRIC_NAMES if n not in fields]
    for row in rows:
        scores = by_id.get(row["id"], {})
        for name in METRIC_NAMES:
            value = scores.get(name)
            row[name] = "" if value is None else round(value, 4)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def apply_ragas(run_dir, progress=print) -> dict:
    run_dir = Path(run_dir)
    config, summary = load_run(run_dir)
    gen_path = run_dir / "generations.jsonl"
    if not gen_path.exists():
        raise RuntimeError("this run has no generations (it was a retrieval-only run)")
    gen_rows = _read_jsonl(gen_path)
    ids = {g["id"] for g in gen_rows}
    items = [i for i in ds.load_items(config["dataset"]["path"]) if i.id in ids]
    if ds.label_hash(items) != config["dataset"]["label_hash"]:
        raise RuntimeError("the dataset labels changed since this run; re-run the evaluation instead of re-scoring")

    result, rows = ragas_runner.score_run(items, gen_rows, progress=progress)
    _write_jsonl(run_dir / "ragas_scores.jsonl", rows)
    _merge_csv(run_dir, rows)
    summary["ragas"] = result
    config["ragas"] = result["judge"]
    save_run(run_dir, config, summary)
    return result


def _count_unvalidated(rows: list, split: str, allow: bool, what: str) -> int:
    unvalidated = [r.id for r in rows if not r.validated]
    if unvalidated and split in ("test", "full") and not allow:
        raise RuntimeError(f"{what}: {len(unvalidated)} unvalidated rows in split '{split}' "
                           f"(e.g. {unvalidated[:5]}); validate them or pass --allow-unvalidated")
    return len(unvalidated)


def apply_tickets(run_dir, allow_unvalidated: bool = False, progress=print) -> dict:
    run_dir = Path(run_dir)
    config, summary = load_run(run_dir)
    scenarios = ticket_eval.select(ticket_eval.load_scenarios(), config["split"])
    n_unvalidated = _count_unvalidated(scenarios, config["split"], allow_unvalidated, "ticket scenarios")
    results = ticket_eval.run_ticket_scenarios(scenarios, progress=progress)
    _write_jsonl(run_dir / "tickets.jsonl", results)
    summary["tickets"] = ticket_eval.ticket_summary(results, n_unvalidated)
    config["ticket_dataset_sha256"] = ds.file_sha256(ticket_eval.SCENARIOS_PATH)
    save_run(run_dir, config, summary)
    return summary["tickets"]


def apply_classification(run_dir, allow_unvalidated: bool = False, progress=print) -> dict:
    run_dir = Path(run_dir)
    config, summary = load_run(run_dir)
    examples = ticket_eval.select(ticket_eval.load_examples(), config["split"])
    n_unvalidated = _count_unvalidated(examples, config["split"], allow_unvalidated, "classification examples")
    results = ticket_eval.run_classification(examples, progress=progress)
    _write_jsonl(run_dir / "classification.jsonl", results)
    summary["classification"] = ticket_eval.classification_summary(results, n_unvalidated)
    config["classification_dataset_sha256"] = ds.file_sha256(ticket_eval.CLASSIFICATION_PATH)
    save_run(run_dir, config, summary)
    return summary["classification"]