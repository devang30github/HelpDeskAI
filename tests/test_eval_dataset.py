import json

import pytest

from app.evaluation import dataset as ds


def _row(**over):
    base = {
        "id": "q001", "split": "dev", "category": "network_wifi", "type": "direct",
        "question": "How do I restart the router?", "answerable": True,
        "reference_answer": "Restart it and wait thirty seconds.", "reference_doc_ids": ["doc-a"],
        "reference_contexts": ["Restart the router and wait thirty seconds."],
        "origin": "human_written", "validated": True,
    }
    base.update(over)
    return base


def _unanswerable(**over):
    base = {
        "id": "q002",
        "type": "unanswerable",
        "answerable": False,
        "reference_answer": None,
        "reference_doc_ids": [],
        "reference_contexts": [],
        "abstention_reason": "Not covered.",
        "category": "general",
    }
    base.update(over)
    return _row(**base)

def _write(tmp_path, rows):
    path = tmp_path / "qa.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_valid_dataset_loads(tmp_path):
    items = ds.load_items(_write(tmp_path, [_row(), _unanswerable()]))
    assert [i.id for i in items] == ["q001", "q002"]
    assert items[1].answerable is False


@pytest.mark.parametrize(
    "row,message",
    [
        (_row(reference_answer=None), "reference_answer"),
        (_row(reference_doc_ids=[]), "reference_doc_ids"),
        (_row(reference_contexts=[]), "reference_contexts"),
        (_row(type="unanswerable"), "answerable must be false"),
        (_row(answerable=False, reference_answer=None, reference_doc_ids=[], reference_contexts=[],
              abstention_reason="x"), "answerable must be false"),
        (_row(category="coffee"), "unknown category"),
        (_row(typo_field=1), "typo_field"),
        (_row(smoke=True, split="test"), "smoke"),
        (_unanswerable(reference_answer="should not be here"), "must not have"),
    ],
)
def test_invalid_rows_are_rejected(tmp_path, row, message):
    with pytest.raises(ds.DatasetError, match=message):
        ds.load_items(_write(tmp_path, [row]))


def test_duplicate_ids_and_bad_json(tmp_path):
    with pytest.raises(ds.DatasetError, match="duplicate id"):
        ds.load_items(_write(tmp_path, [_row(), _row()]))
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ds.DatasetError, match="line 1"):
        ds.load_items(path)


def test_groups_cannot_span_splits(tmp_path):
    rows = [_row(group="g1"), _row(id="q003", split="test", group="g1", type="paraphrase")]
    with pytest.raises(ds.DatasetError, match="spans several splits"):
        ds.load_items(_write(tmp_path, rows))


def test_select_items(tmp_path):
    rows = [_row(smoke=True), _row(id="q003", split="test"), _unanswerable()]
    items = ds.load_items(_write(tmp_path, rows))
    assert [i.id for i in ds.select_items(items, "dev")] == ["q001", "q002"]
    assert [i.id for i in ds.select_items(items, "test")] == ["q003"]
    assert [i.id for i in ds.select_items(items, "smoke")] == ["q001"]
    assert len(ds.select_items(items, "full")) == 3
    with pytest.raises(ds.DatasetError):
        ds.select_items([i for i in items if i.split == "dev"], "test")


def test_validate_against_kb(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "doc-a.txt").write_text(
        "Title: A\nCategory: network_wifi\n\nRestart the router and wait\nthirty seconds. Then test it.",
        encoding="utf-8")
    ok = ds.load_items(_write(tmp_path, [_row()]))
    assert ds.validate_against_kb(ok, kb) == []  # whitespace differences are tolerated

    bad_doc = ds.load_items(_write(tmp_path, [_row(reference_doc_ids=["missing-doc"])]))
    assert any("does not exist" in p for p in ds.validate_against_kb(bad_doc, kb))

    bad_quote = ds.load_items(_write(tmp_path, [_row(reference_contexts=["Unplug it for an hour."])]))
    assert any("not found verbatim" in p for p in ds.validate_against_kb(bad_quote, kb))


def test_kb_pin_check():
    assert ds.check_kb_pin({"kb_files": {"a.txt": "h1"}}, {"a.txt": "h1"}) == []
    assert ds.check_kb_pin({"kb_files": {"a.txt": "h1"}}, {"a.txt": "h2"})
    assert ds.check_kb_pin({"kb_files": {"a.txt": "h1"}}, {"a.txt": "h1", "b.txt": "x"})
    assert ds.check_kb_pin({}, {"a.txt": "h1"})  # no pin at all is a problem


def test_label_hash_ignores_validation_flags_but_not_labels(tmp_path):
    a = ds.load_items(_write(tmp_path, [_row(validated=False)]))
    b = ds.load_items(_write(tmp_path, [_row(validated=True)]))
    c = ds.load_items(_write(tmp_path, [_row(question="How do I power cycle the router?")]))
    assert ds.label_hash(a) == ds.label_hash(b)
    assert ds.label_hash(a) != ds.label_hash(c)


def test_small_dataset_produces_coverage_warnings(tmp_path):
    items = ds.load_items(_write(tmp_path, [_row(), _unanswerable()]))
    warnings = ds.coverage_warnings(items)
    assert any("only 2 questions" in w for w in warnings)
    assert any("unanswerable" in w for w in warnings)