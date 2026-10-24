import pytest

from app.evaluation import metrics as M


def test_hit_and_reciprocal_rank():
    docs = ["b", "a", "c", "a"]
    assert M.hit_at_k(docs, {"a"}, 1) == 0.0
    assert M.hit_at_k(docs, {"a"}, 2) == 1.0
    assert M.reciprocal_rank_at_k(docs, {"a"}, 1) == 0.0
    assert M.reciprocal_rank_at_k(docs, {"a"}, 4) == 0.5  # first relevant chunk at rank 2
    assert M.reciprocal_rank_at_k(docs, {"x"}, 4) == 0.0
    assert M.reciprocal_rank_at_k(["a"], {"a", "b"}, 5) == 1.0
    assert M.hit_at_k([], {"a"}, 3) == 0.0


def test_percentile_is_nearest_rank():
    values = list(range(1, 21))  # 1..20
    assert M.percentile(values, 95) == 19
    assert M.percentile(values, 50) == 10
    assert M.percentile([5.0], 95) == 5.0
    assert M.percentile([], 95) is None


def test_summarize():
    s = M.summarize([10.0, 20.0, 30.0])
    assert s["n"] == 3 and s["median"] == 20.0 and s["p95"] == 30.0 and s["mean"] == 20.0
    assert M.summarize([])["median"] is None


def test_abstention_metrics():
    # (expected_answerable, abstained)
    pairs = [(True, False), (True, True), (False, True), (False, False), (False, True)]
    m = M.abstention_metrics(pairs)
    assert m["n_unanswerable"] == 3 and m["correct_abstentions"] == 2 and m["unsupported_answers"] == 1
    assert m["abstention_recall"] == pytest.approx(2 / 3)
    assert m["false_abstention_rate"] == pytest.approx(1 / 2)
    assert m["abstention_precision"] == pytest.approx(2 / 3)  # 2 correct of 3 abstentions


def test_abstention_metrics_handle_empty_denominators():
    m = M.abstention_metrics([(True, False)])
    assert m["abstention_recall"] is None and m["abstention_precision"] is None
    assert m["false_abstention_rate"] == 0.0


def test_estimate_cost():
    pricing = {"input_per_million_usd": 1.0, "output_per_million_usd": 2.0}
    assert M.estimate_cost(1_000_000, 500_000, pricing) == pytest.approx(2.0)
    assert M.estimate_cost(10, 10, {"input_per_million_usd": None, "output_per_million_usd": 2.0}) is None
    assert M.estimate_cost(10, 10, None) is None