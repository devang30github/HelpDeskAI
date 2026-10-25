"""Compare two evaluation runs (for example V1 baseline vs V2). Pure Python: no app settings needed."""
import csv
import json
from pathlib import Path

from app.evaluation import metrics as M

DIFF_FIELDS = [
    ("retriever", ("retrieval", "retriever")), ("top_k", ("retrieval", "top_k")),
    ("embedding model", ("index", "embedding_model")), ("chunk size", ("index", "chunk_size")),
    ("chunk overlap", ("index", "chunk_overlap")), ("generator model", ("generation", "llm", "model")),
    ("prompt version", ("prompt_version",)),
]
SUMMARY_FIELDS = [
    ("retrieval latency median (ms)", ("latency_ms", "retrieval (eval pass)", "median")),
    ("generation latency median (ms)", ("latency_ms", "generation", "median")),
    ("total latency median (ms)", ("latency_ms", "total", "median")),
    ("total latency p95 (ms)", ("latency_ms", "total", "p95")),
    ("average tokens per question", ("efficiency", "avg_total_tokens")),
    ("abstention recall", ("abstention", "abstention_recall")),
    ("false abstention rate", ("abstention", "false_abstention_rate")),
    ("ticket creation success", ("tickets", "creation_success_rate")),
    ("ticket field completeness", ("tickets", "completeness_rate")),
    ("classification accuracy", ("classification", "accuracy")),
    ("classification macro F1", ("classification", "macro_f1")),
]


def _dig(data, keys):
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def load_run(run_dir) -> dict:
    run_dir = Path(run_dir)
    with open(run_dir / "per_question.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ragas_path = run_dir / "ragas_scores.jsonl"
    ragas = ([json.loads(l) for l in ragas_path.read_text(encoding="utf-8").splitlines() if l.strip()]
             if ragas_path.exists() else [])
    return {
        "dir": run_dir,
        "config": json.loads((run_dir / "config.json").read_text(encoding="utf-8")),
        "summary": json.loads((run_dir / "summary.json").read_text(encoding="utf-8")),
        "rows": rows,
        "ragas": ragas,
    }


def comparability_problems(a: dict, b: dict) -> list[str]:
    ca, cb = a["config"], b["config"]
    problems = []
    if ca["dataset"]["label_hash"] != cb["dataset"]["label_hash"]:
        problems.append("the runs used different dataset labels (label hash differs)")
    if ca["split"] != cb["split"]:
        problems.append(f"different splits: {ca['split']} vs {cb['split']}")
    for name, cfg in (("A", ca), ("B", cb)):
        if not cfg["kb"]["pin_ok"]:
            problems.append(f"run {name} did not match the dataset's knowledge-base pin")
    if ca["kb"].get("indexed_files") != cb["kb"].get("indexed_files"):
        problems.append("different knowledge-base files were indexed")
    ra, rb = ca.get("ragas"), cb.get("ragas")
    if ra and rb and (ra["judge_model"], ra["judge_embedding_model"]) != (rb["judge_model"], rb["judge_embedding_model"]):
        problems.append("the RAGAS judge differs between the runs")
    return problems


def differences(a: dict, b: dict) -> list[tuple]:
    out = []
    for label, keys in DIFF_FIELDS:
        va, vb = _dig(a["config"], keys), _dig(b["config"], keys)
        if va != vb:
            out.append((label, va, vb))
    return out


def _metric_series(run: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    rows = run["rows"]
    for column in (rows[0].keys() if rows else []):
        if column.startswith(("hit@", "rr@")):
            out[column] = {r["id"]: float(r[column]) for r in rows if r.get(column) not in ("", None)}
    out["abstention_correct"] = {
        r["id"]: float((r["abstained"] == "True") == (r["answerable"] == "False"))
        for r in rows if r.get("abstained") in ("True", "False")
    }
    for row in run["ragas"]:
        for name, value in row["scores"].items():
            if value is not None:
                out.setdefault(name, {})[row["id"]] = value
    return {k: v for k, v in out.items() if v}


def _sort_key(name: str):
    group = 0 if name.startswith("hit@") else 1 if name.startswith("rr@") else 2 if name == "abstention_correct" else 3
    k = int(name.split("@")[1]) if "@" in name else 0
    return (group, k, name)


def compare(a: dict, b: dict, n_boot: int = 5000, seed: int = 0) -> dict:
    series_a, series_b = _metric_series(a), _metric_series(b)
    metrics = []
    for name in sorted(set(series_a) & set(series_b), key=_sort_key):
        result = M.paired_bootstrap(series_a[name], series_b[name], n_boot, seed)
        result["metric"] = name
        metrics.append(result)

    headline = a["config"]["retrieval"]["top_k"]
    hit = f"hit@{headline}"
    flips = {"k": headline, "gained": [], "lost": []}
    if hit in series_a and hit in series_b:
        for qid in sorted(set(series_a[hit]) & set(series_b[hit])):
            if series_a[hit][qid] == 0 and series_b[hit][qid] == 1:
                flips["gained"].append(qid)
            elif series_a[hit][qid] == 1 and series_b[hit][qid] == 0:
                flips["lost"].append(qid)
    return {"problems": comparability_problems(a, b), "differences": differences(a, b),
            "metrics": metrics, "flips": flips}


def _f(value, digits=3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _verdict(row: dict) -> str:
    if row["ci_low"] is None:
        return "n/a"
    if row["ci_low"] > 0:
        return "B better"
    if row["ci_high"] < 0:
        return "A better"
    return "no clear difference"


def render(result: dict, a: dict, b: dict) -> str:
    lines = [f"# Comparison: A `{a['config']['run_id']}` (baseline) vs B `{b['config']['run_id']}` (candidate)", ""]
    if result["problems"]:
        lines += ["**NOT COMPARABLE (forced):**"] + [f"- {p}" for p in result["problems"]] + [""]
    if result["differences"]:
        lines += ["## What differs between the runs", ""] + [
            f"- {label}: `{va}` -> `{vb}`" for label, va, vb in result["differences"]] + [""]

    lines += ["## Per-question metrics (paired; diff = B - A; 95% bootstrap interval over questions)", "",
              "| metric | n | A | B | diff | 95% CI | wins/losses/ties | verdict |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in result["metrics"]:
        ci = "n/a" if r["ci_low"] is None else f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        lines.append(f"| {r['metric']} | {r['n']} | {_f(r['mean_a'])} | {_f(r['mean_b'])} | "
                     f"{'n/a' if r['diff'] is None else format(r['diff'], '+.3f')} | {ci} | "
                     f"{r['wins']}/{r['losses']}/{r['ties']} | {_verdict(r)} |")

    flips = result["flips"]
    lines += ["", f"## Questions that changed at Hit@{flips['k']}", "",
              f"- Gained in B: {', '.join(flips['gained']) or 'none'}", f"- Lost in B: {', '.join(flips['lost']) or 'none'}"]

    rows = []
    for label, keys in SUMMARY_FIELDS:
        va, vb = _dig(a["summary"], keys), _dig(b["summary"], keys)
        if va is not None or vb is not None:
            rows.append(f"| {label} | {_f(va, 2)} | {_f(vb, 2)} |")
    if rows:
        lines += ["", "## Run-level metrics (no interval)", "", "| metric | A | B |", "| --- | --- | --- |"] + rows

    lines += ["", "## How to read this", "",
              "- Intervals resample the questions with replacement (paired). With a few dozen questions they are wide: "
              "'no clear difference' means the data cannot separate A from B, not that they are equal.",
              "- RAGAS scores come from an LLM judge, so they carry judge noise and bias in addition to sampling noise.",
              "- Latency depends on network and provider load and comes from single runs; treat small differences as noise."]
    return "\n".join(lines) + "\n"