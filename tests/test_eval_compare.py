import csv
import json

import pytest

from app.evaluation import compare as cmp
import random

def _make_run(tmp_path, name, hits, *, label_hash="L1", kb=None, ragas=None, retriever=None):
    d = tmp_path / name
    d.mkdir()
    config = {"run_id": name, "split": "dev", "dataset": {"label_hash": label_hash},
              "kb": {"pin_ok": True, "indexed_files": kb or {"a.txt": "h"}},
              "retrieval": {"retriever": retriever or name, "top_k": 4}, "index": {}, "generation": None,
              "prompt_version": "v1"}
    (d / "config.json").write_text(json.dumps(config))
    (d / "summary.json").write_text(json.dumps({"latency_ms": {"total": {"median": 100.0}}}))
    with open(d / "per_question.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "answerable", "hit@4", "rr@4", "abstained", "top1_doc"])
        writer.writeheader()
        for i, h in enumerate(hits):
            writer.writerow({"id": f"q{i}", "answerable": True, "hit@4": h, "rr@4": h, "abstained": "", "top1_doc": "d"})
    if ragas:
        (d / "ragas_scores.jsonl").write_text("\n".join(json.dumps(
            {"id": f"q{i}", "scores": {"faithfulness": v}, "errors": {}, "skipped": {}}) for i, v in enumerate(ragas)))
    return cmp.load_run(d)


def test_identical_setups_are_comparable(tmp_path):
    a, b = _make_run(tmp_path, "a", [1, 0]), _make_run(tmp_path, "b", [1, 1])
    assert cmp.comparability_problems(a, b) == []


def test_different_labels_or_kb_are_not_comparable(tmp_path):
    a = _make_run(tmp_path, "a", [1, 0])
    assert any("labels" in p for p in cmp.comparability_problems(a, _make_run(tmp_path, "b", [1, 0], label_hash="L2")))
    assert any("knowledge-base" in p for p in cmp.comparability_problems(a, _make_run(tmp_path, "c", [1, 0], kb={"a.txt": "x"})))


def test_clear_improvement_and_flips(tmp_path):
    a = _make_run(tmp_path, "a", [0] * 10 + [1] * 10)
    b = _make_run(tmp_path, "b", [1] * 20)
    result = cmp.compare(a, b, n_boot=2000, seed=1)
    row = next(r for r in result["metrics"] if r["metric"] == "hit@4")
    assert row["diff"] == pytest.approx(0.5) and row["wins"] == 10 and row["ci_low"] > 0
    assert len(result["flips"]["gained"]) == 10 and result["flips"]["lost"] == []
    text = cmp.render(result, a, b)
    assert "B better" in text and "What differs" in text


def test_no_clear_difference_is_reported_honestly(tmp_path):
    a = _make_run(tmp_path, "a", [1, 0, 1, 0])
    b = _make_run(tmp_path, "b", [0, 1, 1, 0])
    result = cmp.compare(a, b, n_boot=2000, seed=1)
    assert "no clear difference" in cmp.render(result, a, b)


def test_ragas_scores_are_paired_by_question(tmp_path):
    a = _make_run(tmp_path, "a", [1, 1], ragas=[0.5, 0.6])
    b = _make_run(tmp_path, "b", [1, 1], ragas=[0.7, 0.8])
    row = next(r for r in cmp.compare(a, b, n_boot=500)["metrics"] if r["metric"] == "faithfulness")
    assert row["diff"] == pytest.approx(0.2) and row["n"] == 2