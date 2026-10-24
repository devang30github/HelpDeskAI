"""Evaluation runner: a deterministic retrieval pass, an optional generation pass,
and run artifacts written to disk."""
import csv
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from app.evaluation import metrics as M
from app.evaluation.dataset import EvalItem
from app.evaluation.report import write_report
from app.rag.generation import LLMError

logger = logging.getLogger(__name__)


def retrieval_pass(items: list[EvalItem], retriever, eval_k: int) -> list[dict]:
    retriever.retrieve("warm-up query", 1)  # excluded from timing: loads the model if it is not loaded yet
    rows = []
    for item in items:
        start = time.perf_counter()
        chunks = retriever.retrieve(item.question, eval_k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        rows.append({
            "id": item.id,
            "retrieval_ms": elapsed_ms,
            "retrieved": [
                {"rank": c.rank, "chunk_id": c.chunk_id, "doc_id": c.doc_id,
                 "source_name": c.source_name, "score": c.score}
                for c in chunks
            ],
        })
    return rows


def _aggregate(per_item: dict, ids: list[str], k_values: list[int]) -> dict:
    ids = [i for i in ids if i in per_item]
    if not ids:
        return {"n": 0}
    out = {"n": len(ids)}
    for k in k_values:
        out[f"hit@{k}"] = mean(per_item[i][f"hit@{k}"] for i in ids)
        out[f"mrr@{k}"] = mean(per_item[i][f"rr@{k}"] for i in ids)
    return out


def retrieval_summary(items, rows, k_values, headline_k) -> tuple[dict, dict]:
    by_id = {r["id"]: r for r in rows}
    per_item: dict[str, dict] = {}
    for item in items:
        if not item.answerable:
            continue
        doc_ids = [c["doc_id"] for c in by_id[item.id]["retrieved"]]
        relevant = set(item.reference_doc_ids)
        per_item[item.id] = {}
        for k in k_values:
            per_item[item.id][f"hit@{k}"] = M.hit_at_k(doc_ids, relevant, k)
            per_item[item.id][f"rr@{k}"] = M.reciprocal_rank_at_k(doc_ids, relevant, k)

    answerable = [i for i in items if i.answerable]
    summary = {
        "k_values": k_values,
        "headline_k": headline_k,
        "n": len(answerable),
        "overall": _aggregate(per_item, [i.id for i in answerable], k_values),
        "by_type": {t: _aggregate(per_item, [i.id for i in answerable if i.type == t], [headline_k])
                    for t in sorted({i.type for i in answerable})},
        "by_category": {c: _aggregate(per_item, [i.id for i in answerable if i.category == c], [headline_k])
                        for c in sorted({i.category for i in answerable})},
    }
    return summary, per_item


def generation_pass(items: list[EvalItem], pipeline, sleep: float = 0.0, progress=None) -> list[dict]:
    rows = []
    for index, item in enumerate(items, start=1):
        if progress:
            progress(f"[{index}/{len(items)}] {item.id}")
        row = {"id": item.id, "question": item.question}
        try:
            result = pipeline.answer(item.question)
        except LLMError as e:  # recorded, never silently dropped
            logger.warning("generation failed for %s: %s", item.id, e)
            row["error"] = str(e)
            rows.append(row)
            continue
        row.update(
            error=None,
            answer=result.answer,
            answerable=result.answerable,
            sources=[asdict(s) for s in result.sources],
            retrieved_doc_ids=[c.doc_id for c in result.retrieved],
            contexts=result.contexts,
            timings=result.timings,
            usage=result.usage,
        )
        rows.append(row)
        if sleep:
            time.sleep(sleep)
    return rows


def _generation_summary(items, gen_rows, retr_rows, pricing) -> dict:
    by_id = {i.id: i for i in items}
    ok = [r for r in gen_rows if not r.get("error")]
    failures = [{"id": r["id"], "error": r["error"]} for r in gen_rows if r.get("error")]
    abstention = M.abstention_metrics([(by_id[r["id"]].answerable, not r["answerable"]) for r in ok])

    prompt_tokens = sum(r["usage"]["prompt_tokens"] for r in ok)
    completion_tokens = sum(r["usage"]["completion_tokens"] for r in ok)
    total_tokens = sum(r["usage"]["total_tokens"] for r in ok)
    review = {
        "unsupported_answers": [
            {"id": r["id"], "question": r["question"], "answer": r["answer"][:300]}
            for r in ok if not by_id[r["id"]].answerable and r["answerable"]
        ],
        "false_abstentions": [
            {"id": r["id"], "question": r["question"], "answer": r["answer"][:300]}
            for r in ok if by_id[r["id"]].answerable and not r["answerable"]
        ],
    }
    return {
        "generation": {"n_attempted": len(gen_rows), "n_ok": len(ok), "n_failed": len(failures), "failures": failures},
        "abstention": abstention,
        "latency_ms": {
            "retrieval (eval pass)": M.summarize([r["retrieval_ms"] for r in retr_rows]),
            "retrieval (in pipeline)": M.summarize([r["timings"]["retrieval_ms"] for r in ok]),
            "generation": M.summarize([r["timings"]["generation_ms"] for r in ok]),
            "total": M.summarize([r["timings"]["total_ms"] for r in ok]),
        },
        "efficiency": {
            "n_questions": len(ok),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "avg_total_tokens": total_tokens / len(ok) if ok else None,
            "pricing": pricing,
            "estimated_cost_usd": M.estimate_cost(prompt_tokens, completion_tokens, pricing),
        },
        "manual_review": review,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    path.write_text(body + ("\n" if rows else ""), encoding="utf-8")


def _write_table(path, items, retr_rows, per_item, gen_rows, k_values) -> None:
    retr = {r["id"]: r for r in retr_rows}
    gen = {r["id"]: r for r in gen_rows}
    columns = ["id", "split", "category", "type", "answerable", "validated", "retrieval_ms_eval",
               "top1_doc", "top1_score"]
    columns += [f"hit@{k}" for k in k_values] + [f"rr@{k}" for k in k_values]
    columns += ["generation_error", "abstained", "retrieval_ms", "generation_ms", "total_ms",
                "prompt_tokens", "completion_tokens"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            r = retr[item.id]
            top = r["retrieved"][0] if r["retrieved"] else None
            row = {"id": item.id, "split": item.split, "category": item.category, "type": item.type,
                   "answerable": item.answerable, "validated": item.validated,
                   "retrieval_ms_eval": round(r["retrieval_ms"], 2),
                   "top1_doc": top["doc_id"] if top else "", "top1_score": round(top["score"], 4) if top else ""}
            for k in k_values:
                if item.id in per_item:
                    row[f"hit@{k}"] = per_item[item.id][f"hit@{k}"]
                    row[f"rr@{k}"] = round(per_item[item.id][f"rr@{k}"], 4)
            g = gen.get(item.id)
            if g:
                row["generation_error"] = g.get("error") or ""
                if not g.get("error"):
                    row.update(abstained=not g["answerable"], retrieval_ms=round(g["timings"]["retrieval_ms"], 1),
                               generation_ms=round(g["timings"]["generation_ms"], 1),
                               total_ms=round(g["timings"]["total_ms"], 1),
                               prompt_tokens=g["usage"]["prompt_tokens"],
                               completion_tokens=g["usage"]["completion_tokens"])
            writer.writerow(row)


def run_evaluation(
    *, items, retriever, pipeline, out_dir: Path, run_config: dict, k_values: list[int],
    eval_k: int, headline_k: int, pricing: dict | None = None, sleep: float = 0.0, progress=None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    retr_rows = retrieval_pass(items, retriever, eval_k)
    retr_summary, per_item = retrieval_summary(items, retr_rows, k_values, headline_k)

    gen_rows: list[dict] = []
    summary: dict = {
        "run_id": run_config["run_id"],
        "counts": {
            "items": len(items),
            "answerable": sum(i.answerable for i in items),
            "unanswerable": sum(not i.answerable for i in items),
            "unvalidated": sum(not i.validated for i in items),
        },
        "retrieval": retr_summary,
    }
    if pipeline is not None:
        gen_rows = generation_pass(items, pipeline, sleep=sleep, progress=progress)
        summary.update(_generation_summary(items, gen_rows, retr_rows, pricing))
    else:
        summary["latency_ms"] = {"retrieval (eval pass)": M.summarize([r["retrieval_ms"] for r in retr_rows])}

    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_jsonl(out_dir / "retrieval.jsonl", retr_rows)
    if pipeline is not None:
        _write_jsonl(out_dir / "generations.jsonl", gen_rows)
    _write_table(out_dir / "per_question.csv", items, retr_rows, per_item, gen_rows, k_values)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, run_config, out_dir / "report.md")
    return summary