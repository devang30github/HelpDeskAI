"""Human-readable report for one run."""
from pathlib import Path

DEFINITIONS = """\
- **Hit@k**: share of answerable questions where at least one of the top-k retrieved chunks comes from a reference document. Denominator: answerable questions.
- **MRR@k**: mean over answerable questions of 1/rank of the first retrieved chunk from a reference document within the top k (0 if none). Ranks count chunks, not unique documents.
- **Abstention**: the pipeline flag `answerable=false` (the model emitted INSUFFICIENT_EVIDENCE, or retrieval returned nothing). Chat intent routing is not part of this evaluation.
- **Abstention recall**: correct abstentions / unanswerable questions (failed generations excluded). **False abstention rate**: abstentions on answerable questions / answerable questions. **Abstention precision**: correct abstentions / all abstentions.
- **Latency**: `retrieval (eval pass)` is a separate warm pass over all questions; generation and total come from successful LLM calls only. p95 uses the nearest-rank method. LLM latency includes network variance.
- **Tokens**: provider-reported usage of generation calls only (judge calls are not included). **Cost** is shown only when prices are configured; API responses do not include prices.
- **Failures**: a failed generation is listed below with its reason and excluded from generation metrics. Retrieval metrics do not depend on the LLM.
- Metrics from an LLM judge (RAGAS, added in the next stage) are judge scores, not independently verified ground truth.
"""


def _f(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(summary: dict, run_config: dict, path: Path) -> None:
    counts, retr = summary["counts"], summary["retrieval"]
    gen_cfg = run_config.get("generation") or {}
    lines = [
        f"# Evaluation report: {run_config['run_id']}", "",
        f"- Split: **{run_config['split']}** ({counts['items']} questions: {counts['answerable']} answerable, "
        f"{counts['unanswerable']} unanswerable; {counts['unvalidated']} not yet hand-validated)",
        f"- Dataset: {run_config['dataset']['version']} (labels `{run_config['dataset']['label_hash'][:12]}`, "
        f"file `{run_config['dataset']['file_sha256'][:12]}`)",
        f"- Knowledge base pin: {'matches' if run_config['kb']['pin_ok'] else '**MISMATCH (not comparable)**'}",
        f"- Retriever: `{run_config['retrieval']['retriever']}`, top_k={run_config['retrieval']['top_k']}, "
        f"embedding `{run_config['index'].get('embedding_model')}`, chunks {run_config['index'].get('chunk_size')}/"
        f"{run_config['index'].get('chunk_overlap')}",
        f"- Generator: `{(gen_cfg.get('llm') or {}).get('model', 'not run (retrieval only)')}`, prompt `{run_config['prompt_version']}`",
        f"- Git: `{(run_config['environment']['git'].get('commit') or 'unknown')[:10]}`"
        f"{' (uncommitted changes)' if run_config['environment']['git'].get('dirty') else ''}",
        "", f"## Retrieval (document level, N = {retr['n']} answerable questions)", "",
    ]
    overall = retr["overall"]
    lines.append(_table(["k", "Hit@k", "MRR@k"],
                        [[k, _f(overall.get(f"hit@{k}")), _f(overall.get(f"mrr@{k}"))] for k in retr["k_values"]]))

    hk = retr["headline_k"]
    for title, key in (("question type", "by_type"), ("category", "by_category")):
        lines += ["", f"### By {title} (k = {hk})", ""]
        lines.append(_table([title, "n", f"Hit@{hk}", f"MRR@{hk}"],
                            [[name, v["n"], _f(v.get(f"hit@{hk}")), _f(v.get(f"mrr@{hk}"))]
                             for name, v in sorted(retr[key].items())]))

    gen = summary.get("generation")
    if gen:
        ab = summary["abstention"]
        lines += [
            "", "## Generation and abstention", "",
            f"- Attempted {gen['n_attempted']}, succeeded {gen['n_ok']}, failed {gen['n_failed']}",
            f"- Abstention recall: {_f(ab['abstention_recall'])} ({ab['correct_abstentions']}/{ab['n_unanswerable']} unanswerable)",
            f"- False abstention rate: {_f(ab['false_abstention_rate'])} ({ab['false_abstentions']}/{ab['n_answerable']} answerable)",
            f"- Abstention precision: {_f(ab['abstention_precision'])} ({ab['correct_abstentions']}/{ab['total_abstentions']} abstentions)",
        ]
        lat = summary["latency_ms"]
        lines += ["", "## Latency (ms)", ""]
        lines.append(_table(["stage", "n", "median", "p95"],
                            [[name, v["n"], _f(v["median"], 1), _f(v["p95"], 1)] for name, v in lat.items()]))
        eff = summary["efficiency"]
        cost = "n/a (prices not configured)" if eff["estimated_cost_usd"] is None else f"${eff['estimated_cost_usd']:.4f}"
        lines += ["", "## Efficiency", "",
                  f"- Tokens: {eff['total_tokens']} total ({eff['prompt_tokens']} prompt, {eff['completion_tokens']} completion); "
                  f"average {_f(eff['avg_total_tokens'], 0)} per question",
                  f"- Estimated generation cost: {cost}"]
    else:
        lat = summary["latency_ms"]
        lines += ["", "## Latency (ms)", "",
                  _table(["stage", "n", "median", "p95"],
                         [[name, v["n"], _f(v["median"], 1), _f(v["p95"], 1)] for name, v in lat.items()]),
                  "", "_Retrieval-only run: no generation metrics._"]

    if gen and gen["failures"]:
        lines += ["", "## Failed generations", ""] + [f"- `{f['id']}`: {f['error']}" for f in gen["failures"]]
    review = summary.get("manual_review") or {}
    if review.get("unsupported_answers"):
        lines += ["", "## Review manually: answered although the knowledge base cannot answer", ""]
        lines += [f"- `{r['id']}` {r['question']}\n  > {r['answer']}" for r in review["unsupported_answers"][:20]]
    if review.get("false_abstentions"):
        lines += ["", "## Review manually: abstained although the knowledge base can answer", ""]
        lines += [f"- `{r['id']}` {r['question']}\n  > {r['answer']}" for r in review["false_abstentions"][:20]]

    lines += ["", "## Metric definitions and denominators", "", DEFINITIONS]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")