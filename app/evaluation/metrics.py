"""Pure metric functions (no I/O, no LLM) so they are fully unit-testable."""
import math
from statistics import mean, median


def hit_at_k(retrieved_doc_ids: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(d in relevant for d in retrieved_doc_ids[:k]) else 0.0


def reciprocal_rank_at_k(retrieved_doc_ids: list[str], relevant: set[str], k: int) -> float:
    """1 / rank of the first retrieved chunk from a relevant document (0 if none in the top k).
    Ranks count chunks, not unique documents."""
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p / 100 * len(ordered))) - 1]


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "median": None, "p95": None, "mean": None, "max": None}
    return {"n": len(values), "median": median(values), "p95": percentile(values, 95),
            "mean": mean(values), "max": max(values)}


def abstention_metrics(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs are (expected_answerable, abstained)."""
    unanswerable = [abstained for expected, abstained in pairs if not expected]
    answerable = [abstained for expected, abstained in pairs if expected]
    correct = sum(unanswerable)
    false_abstentions = sum(answerable)
    total_abstentions = correct + false_abstentions
    return {
        "n_unanswerable": len(unanswerable),
        "n_answerable": len(answerable),
        "correct_abstentions": correct,
        "unsupported_answers": len(unanswerable) - correct,
        "false_abstentions": false_abstentions,
        "total_abstentions": total_abstentions,
        "abstention_recall": correct / len(unanswerable) if unanswerable else None,
        "false_abstention_rate": false_abstentions / len(answerable) if answerable else None,
        "abstention_precision": correct / total_abstentions if total_abstentions else None,
    }


def estimate_cost(prompt_tokens: int, completion_tokens: int, pricing: dict | None) -> float | None:
    pricing = pricing or {}
    price_in, price_out = pricing.get("input_per_million_usd"), pricing.get("output_per_million_usd")
    if price_in is None or price_out is None:
        return None
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000