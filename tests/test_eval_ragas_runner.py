import asyncio
from types import SimpleNamespace

import pytest

from app.evaluation.dataset import EvalItem
from app.evaluation.ragas_runner import plan_for, ragas_summary, score_generations
from app.evaluation.ragas_setup import METRIC_NAMES


def _item(id_, answerable=True):
    if answerable:
        return EvalItem(id=id_, split="dev", category="network_wifi", type="direct", question=f"Question {id_} about wifi?",
                        answerable=True, reference_answer="Ref answer.", reference_doc_ids=["d"],
                        reference_contexts=["ctx"], origin="human_written", validated=True)
    return EvalItem(id=id_, split="dev", category="general", type="unanswerable", question=f"Question {id_} about HR?",
                    answerable=False, abstention_reason="Not covered.", origin="human_written", validated=True)


def _gen(id_, answerable=True, error=None, contexts=("c1",)):
    if error:
        return {"id": id_, "error": error}
    return {"id": id_, "error": None, "answer": "An answer [1].", "answerable": answerable, "contexts": list(contexts)}


class FakeMetric:
    def __init__(self, value=0.5, error=None):
        self.value, self.error, self.calls = value, error, []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(value=self.value)


def _metrics(**overrides):
    return {name: overrides.get(name, FakeMetric()) for name in METRIC_NAMES}


def test_plan_defines_the_denominators():
    answered = plan_for(_item("q1"), _gen("q1"))
    assert all(kwargs is not None for kwargs, _ in answered.values())

    abstained = plan_for(_item("q2"), _gen("q2", answerable=False))
    assert abstained["faithfulness"][0] is None and "abstained" in abstained["faithfulness"][1]
    assert abstained["answer_relevancy"][0] is None
    assert abstained["context_precision"][0] is not None and abstained["context_recall"][0] is not None

    assert all(k is None for k, _ in plan_for(_item("q3", answerable=False), _gen("q3", answerable=False)).values())
    assert all(r == "generation failed" for _, r in plan_for(_item("q4"), _gen("q4", error="boom")).values())
    assert plan_for(_item("q5"), _gen("q5", contexts=()))["context_recall"][1] == "no retrieved contexts"


def test_scores_failures_and_nan_are_recorded():
    items = [_item("q1"), _item("q2", answerable=False), _item("q3")]
    gens = [_gen("q1"), _gen("q2", answerable=False), _gen("q3", answerable=False)]
    metrics = _metrics(faithfulness=FakeMetric(error=ValueError("judge exploded")),
                       context_recall=FakeMetric(value=float("nan")))
    rows = asyncio.run(score_generations(items, gens, metrics, concurrency=2))
    by_id = {r["id"]: r for r in rows}

    assert by_id["q1"]["scores"]["answer_relevancy"] == 0.5
    assert "ValueError" in by_id["q1"]["errors"]["faithfulness"]
    assert by_id["q1"]["errors"]["context_recall"] == "metric returned NaN"
    assert set(by_id["q2"]["skipped"]) == set(METRIC_NAMES)  # unanswerable: no RAGAS scoring
    assert set(by_id["q3"]["scores"]) == {"context_precision", "context_recall"}  # abstained on an answerable question

    summary = ragas_summary(rows)
    assert summary["faithfulness"]["n_scored"] == 0 and summary["faithfulness"]["n_failed"] == 1
    assert summary["faithfulness"]["n_skipped"] == 2
    assert summary["context_precision"]["n_scored"] == 2 and summary["context_precision"]["mean"] == 0.5
    assert summary["context_recall"]["n_failed"] == 2 and summary["context_recall"]["mean"] is None

def test_rate_limited_metric_fails_over_to_the_next_key():
    pool = [_metrics(faithfulness=FakeMetric(error=RuntimeError("RateLimitError: 429 too many requests"))), _metrics()]
    rows = asyncio.run(score_generations([_item("q1")], [_gen("q1")], pool, concurrency=1))
    assert rows[0]["scores"]["faithfulness"] == 0.5
    assert "faithfulness" not in rows[0]["errors"]

def test_score_run_dependencies_are_importable():
    from app.evaluation import ragas_runner

    assert callable(ragas_runner.build_metrics_pool) and ragas_runner.itertools is not None