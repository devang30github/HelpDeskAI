import csv
import json

import pytest

from app.evaluation.dataset import EvalItem
from app.evaluation.runner import run_evaluation
from app.rag.generation import LLMError
from app.rag.pipeline import RAGResult, SourceRef
from app.rag.types import RetrievedChunk


def _answerable(id_, doc):
    return EvalItem(id=id_, split="dev", category="network_wifi", type="direct", question=f"Question {id_} about wifi?",
                    answerable=True, reference_answer="Ref.", reference_doc_ids=[doc], reference_contexts=["ctx"],
                    origin="human_written", validated=True)


def _unanswerable(id_):
    return EvalItem(id=id_, split="dev", category="general", type="unanswerable", question=f"Question {id_} about HR?",
                    answerable=False, abstention_reason="Not covered.", origin="human_written", validated=True)


class FakeRetriever:
    name = "fake"

    def __init__(self, docs_by_question):
        self.docs_by_question = docs_by_question

    def retrieve(self, query, k):
        docs = self.docs_by_question.get(query, ["doc-x"])[:k]
        return [RetrievedChunk(f"{d}::chunk-000", d, f"{d}.txt", f"text of {d}", 0.9 - i * 0.1, i + 1, {})
                for i, d in enumerate(docs)]


class FakePipeline:
    def __init__(self, behaviour):
        self.behaviour = behaviour  # question -> "answer" | "abstain" | "error"

    def config(self):
        return {"pipeline": "fake", "llm": {"model": "fake-model"}}

    def answer(self, question, history=None):
        mode = self.behaviour[question]
        if mode == "error":
            raise LLMError("provider rate limit reached")
        chunk = RetrievedChunk("doc-b::chunk-000", "doc-b", "doc-b.txt", "text", 0.8, 1, {})
        return RAGResult(
            question=question, answer="answer text [1]" if mode == "answer" else "Not in the KB.",
            answerable=mode == "answer",
            sources=[SourceRef(1, chunk.chunk_id, "doc-b", "doc-b.txt", "B", 0.8, 1, mode == "answer")],
            retrieved=[chunk],
            timings={"retrieval_ms": 5.0, "generation_ms": 100.0, "total_ms": 105.0},
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            config={},
        )


@pytest.fixture
def run(tmp_path):
    items = [_answerable("q1", "doc-a"), _answerable("q2", "doc-d"), _unanswerable("q3"),
             _unanswerable("q4"), _answerable("q5", "doc-e")]
    q = {i.id: i.question for i in items}
    retriever = FakeRetriever({q["q1"]: ["doc-b", "doc-a", "doc-c"], q["q2"]: ["doc-d", "doc-b"],
                               q["q3"]: ["doc-b"], q["q4"]: ["doc-b"], q["q5"]: ["doc-e"]})
    pipeline = FakePipeline({q["q1"]: "answer", q["q2"]: "answer", q["q3"]: "abstain",
                             q["q4"]: "answer", q["q5"]: "error"})
    config = {
        "run_id": "test-run", "split": "dev",
        "dataset": {"version": "1.0", "label_hash": "a" * 64, "file_sha256": "b" * 64},
        "kb": {"pin_ok": True}, "index": {"embedding_model": "m", "chunk_size": 800, "chunk_overlap": 120},
        "retrieval": {"retriever": "fake", "top_k": 3}, "generation": pipeline.config(),
        "prompt_version": "v1", "environment": {"git": {"commit": None, "dirty": None}},
    }
    out = tmp_path / "run"
    summary = run_evaluation(
        items=items, retriever=retriever, pipeline=pipeline, out_dir=out, run_config=config,
        k_values=[1, 3], eval_k=3, headline_k=3,
        pricing={"input_per_million_usd": 1.0, "output_per_million_usd": 2.0},
    )
    return summary, out


def test_retrieval_metrics(run):
    summary, _ = run
    r = summary["retrieval"]
    assert r["n"] == 3  # answerable questions only, regardless of generation failures
    assert r["overall"]["hit@1"] == pytest.approx(2 / 3)
    assert r["overall"]["hit@3"] == pytest.approx(1.0)
    assert r["overall"]["mrr@3"] == pytest.approx((0.5 + 1 + 1) / 3)
    assert r["by_type"]["direct"]["n"] == 3


def test_abstention_and_failures(run):
    summary, _ = run
    gen = summary["generation"]
    assert (gen["n_attempted"], gen["n_ok"], gen["n_failed"]) == (5, 4, 1)
    assert gen["failures"] == [{"id": "q5", "error": "provider rate limit reached"}]
    ab = summary["abstention"]
    assert ab["n_unanswerable"] == 2 and ab["correct_abstentions"] == 1 and ab["abstention_recall"] == 0.5
    assert ab["false_abstention_rate"] == 0.0
    assert [r["id"] for r in summary["manual_review"]["unsupported_answers"]] == ["q4"]


def test_latency_tokens_and_cost(run):
    summary, _ = run
    assert summary["latency_ms"]["generation"]["median"] == 100.0
    eff = summary["efficiency"]
    assert (eff["prompt_tokens"], eff["completion_tokens"], eff["total_tokens"]) == (40, 20, 60)
    assert eff["estimated_cost_usd"] == pytest.approx(0.00008)


def test_artifacts_are_written(run):
    _, out = run
    for name in ("config.json", "retrieval.jsonl", "generations.jsonl", "per_question.csv", "summary.json", "report.md"):
        assert (out / name).exists(), name
    rows = list(csv.DictReader(open(out / "per_question.csv", encoding="utf-8")))
    assert len(rows) == 5 and rows[4]["generation_error"] == "provider rate limit reached"
    assert json.loads((out / "summary.json").read_text())["counts"]["answerable"] == 3
    report = (out / "report.md").read_text()
    assert "## Retrieval" in report and "Failed generations" in report and "q5" in report


def test_retrieval_only_run_makes_no_generation_artifacts(tmp_path):
    items = [_answerable("q1", "doc-a")]
    retriever = FakeRetriever({items[0].question: ["doc-a"]})
    config = {
        "run_id": "ro", "split": "dev", "dataset": {"version": "1.0", "label_hash": "a" * 64, "file_sha256": "b" * 64},
        "kb": {"pin_ok": True}, "index": {}, "retrieval": {"retriever": "fake", "top_k": 3},
        "generation": None, "prompt_version": "v1", "environment": {"git": {}},
    }
    summary = run_evaluation(items=items, retriever=retriever, pipeline=None, out_dir=tmp_path / "ro",
                             run_config=config, k_values=[1, 3], eval_k=3, headline_k=3)
    assert "generation" not in summary and summary["retrieval"]["overall"]["hit@1"] == 1.0
    assert not (tmp_path / "ro" / "generations.jsonl").exists()
    assert "Retrieval-only run" in (tmp_path / "ro" / "report.md").read_text()