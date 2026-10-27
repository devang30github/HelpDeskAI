"""RAGAS scoring over saved generations: per question, concurrency-limited,
with every failure and skip recorded (nothing is silently dropped)."""
import asyncio
import logging
import math
from statistics import mean, median
import itertools
from app.core.config import get_settings
from app.evaluation.dataset import EvalItem
from app.evaluation.ragas_setup import METRIC_NAMES, build_metrics_pool, describe_judge
logger = logging.getLogger(__name__)


def plan_for(item: EvalItem, gen_row: dict) -> dict[str, tuple[dict | None, str | None]]:
    """metric -> (kwargs for ascore, skip reason). This defines each metric's denominator."""
    def skipped(reason: str) -> dict:
        return {name: (None, reason) for name in METRIC_NAMES}

    if gen_row.get("error"):
        return skipped("generation failed")
    if not item.answerable:
        return skipped("unanswerable question (covered by the abstention metrics)")

    contexts = gen_row.get("contexts") or []
    answered = bool(gen_row.get("answerable"))
    response = gen_row.get("answer", "")
    plan: dict[str, tuple[dict | None, str | None]] = {}

    abstained = "system abstained (counted in the false-abstention rate)"
    no_context = "no retrieved contexts"
    gen_kwargs = dict(user_input=item.question, response=response, retrieved_contexts=contexts)
    plan["faithfulness"] = (gen_kwargs, None) if answered and contexts else (None, abstained if not answered else no_context)
    plan["answer_relevancy"] = (dict(user_input=item.question, response=response), None) if answered else (None, abstained)

    retr_kwargs = dict(user_input=item.question, retrieved_contexts=contexts, reference=item.reference_answer)
    for name in ("context_precision", "context_recall"):
        plan[name] = (retr_kwargs, None) if contexts else (None, no_context)
    return plan

def _is_rate_limit(error: str | None) -> bool:
    return bool(error) and ("ratelimit" in error.lower().replace(" ", "") or "429" in error)


async def score_generations(items, gen_rows, metrics, concurrency: int = 4, progress=None) -> list[dict]:
    """`metrics` is one metrics dict, or a list of them (one per API key)."""
    pool = metrics if isinstance(metrics, list) else [metrics]
    by_item = {i.id: i for i in items}
    semaphore = asyncio.Semaphore(max(1, concurrency))
    rows: dict[str, dict] = {}
    turn = itertools.count()
    done = 0

    async def run(row_id: str, name: str, kwargs: dict, total: int):
        nonlocal done
        async with semaphore:
            first = next(turn)  # round robin over the judge clients
            value, error = None, None
            for attempt in range(min(len(pool), 3)):  # on a rate limit, try the next key
                value, error = await _score_one(pool[(first + attempt) % len(pool)][name], kwargs)
                if error is None or not _is_rate_limit(error):
                    break
        rows[row_id]["scores"][name] = value
        if error:
            rows[row_id]["errors"][name] = error
        done += 1
        if progress and (done % 10 == 0 or done == total):
            progress(f"  RAGAS {done}/{total} metric evaluations")

    planned = []
    for gen_row in gen_rows:
        item = by_item.get(gen_row["id"])
        if item is None:
            continue
        rows[gen_row["id"]] = {"id": gen_row["id"], "scores": {}, "errors": {}, "skipped": {}}
        for name, (kwargs, reason) in plan_for(item, gen_row).items():
            if kwargs is None:
                rows[gen_row["id"]]["skipped"][name] = reason
            else:
                planned.append((gen_row["id"], name, kwargs))
    await asyncio.gather(*(run(rid, name, kwargs, len(planned)) for rid, name, kwargs in planned))
    return [rows[g["id"]] for g in gen_rows if g["id"] in rows]


def score_run(items, gen_rows, concurrency: int | None = None, progress=None) -> tuple[dict, list[dict]]:
    settings = get_settings()
    metrics = build_metrics_pool(settings)
    concurrency = concurrency or settings.ragas_concurrency
    rows = asyncio.run(score_generations(items, gen_rows, metrics, concurrency, progress))
    return {"judge": describe_judge(settings), "concurrency": concurrency, "metrics": ragas_summary(rows)}, rows

async def _score_one(metric, kwargs: dict) -> tuple[float | None, str | None]:
    try:
        result = await metric.ascore(**kwargs)
        value = float(getattr(result, "value", result))
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:300]}"
    if math.isnan(value):
        return None, "metric returned NaN"
    return value, None


def ragas_summary(rows: list[dict]) -> dict:
    out = {}
    for name in METRIC_NAMES:
        scored = [r["scores"][name] for r in rows if r["scores"].get(name) is not None]
        failures = [{"id": r["id"], "error": r["errors"][name]} for r in rows if name in r["errors"]]
        out[name] = {
            "n_scored": len(scored),
            "n_failed": len(failures),
            "n_skipped": sum(1 for r in rows if name in r["skipped"]),
            "mean": mean(scored) if scored else None,
            "median": median(scored) if scored else None,
            "failures": failures[:20],
        }
    return out

def score_run(items, gen_rows, concurrency: int | None = None, progress=None) -> tuple[dict, list[dict]]:
    settings = get_settings()
    metrics = build_metrics_pool(settings)
    concurrency = concurrency or settings.ragas_concurrency
    rows = asyncio.run(score_generations(items, gen_rows, metrics, concurrency, progress))
    return {"judge": describe_judge(settings), "concurrency": concurrency, "metrics": ragas_summary(rows)}, rows


