"""Pure metric functions (no I/O, no LLM) so they are fully unit-testable."""
import math 
import random
from collections import Counter
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

def classification_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """Accuracy plus per-class precision/recall/F1. Precision, recall and F1 are 0 when undefined.
    Macro averages cover only classes that have at least one true example."""
    n = len(y_true)
    per_class = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    scored = [v for v in per_class.values() if v["support"] > 0]
    confusions = Counter(f"{t} -> {p}" for t, p in zip(y_true, y_pred) if t != p)
    return {
        "n": n,
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / n if n else None,
        "per_class": per_class,
        "macro_precision": mean(v["precision"] for v in scored) if scored else None,
        "macro_recall": mean(v["recall"] for v in scored) if scored else None,
        "macro_f1": mean(v["f1"] for v in scored) if scored else None,
        "confusions": dict(confusions.most_common()),
    }


def paired_bootstrap(a: dict[str, float], b: dict[str, float], n_boot: int = 5000, seed: int = 0) -> dict:
    """Paired bootstrap over questions present in both runs. diff = B - A."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"n": 0, "mean_a": None, "mean_b": None, "diff": None, "ci_low": None, "ci_high": None,
                "wins": 0, "losses": 0, "ties": 0}
    diffs = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(n_boot))
    return {
        "n": len(ids),
        "mean_a": mean(a[i] for i in ids),
        "mean_b": mean(b[i] for i in ids),
        "diff": mean(diffs),
        "ci_low": means[int(0.025 * n_boot)],
        "ci_high": means[int(0.975 * n_boot) - 1],
        "wins": sum(d > 0 for d in diffs),
        "losses": sum(d < 0 for d in diffs),
        "ties": sum(d == 0 for d in diffs),
    }