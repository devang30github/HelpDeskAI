import json

import pytest

from app.evaluation import ticket_eval
from app.rag.generation import LLMError, LLMResult


class StubLLM:
    """Returns the same drafting JSON for every call."""

    def __init__(self, draft):
        self.draft = draft

    def complete(self, messages, max_tokens=None):
        return LLMResult(json.dumps(self.draft), "stub", 1, 1, 2, 1.0, "stop")


def _scenario(id_="t01", category="network_wifi"):
    return ticket_eval.Scenario(id=id_, split="dev", expected_category=category, validated=True,
                                problem="VPN times out whenever I work from home and restarting did not help.")


FULL = {"title": "VPN timeout from home", "description": "VPN times out whenever I work from home.",
        "category": "network_wifi", "priority": "high"}


def test_full_draft_creates_a_complete_ticket_in_two_turns():
    llm = StubLLM(FULL)
    [r] = ticket_eval.run_ticket_scenarios([_scenario()], llm_factory=lambda: llm)
    assert r["created"] and r["turns"] == 2 and r["complete"] is True and r["missing_fields"] == []
    assert r["category_matches"] is True and r["ticket_id"].startswith("HD-")


def test_missing_category_is_collected_from_the_simulated_employee():
    llm = StubLLM({**FULL, "category": None})
    [r] = ticket_eval.run_ticket_scenarios([_scenario()], llm_factory=lambda: llm)
    assert r["created"] and r["turns"] == 3 and r["category"] == "network_wifi"


def test_empty_draft_needs_description_and_category():
    llm = StubLLM({"title": None, "description": None, "category": None, "priority": None})
    [r] = ticket_eval.run_ticket_scenarios([_scenario()], llm_factory=lambda: llm)
    assert r["created"] and r["turns"] == 4 and r["priority"] == "medium"  # default priority applied


def test_wrong_suggested_category_is_reported_not_hidden():
    llm = StubLLM({**FULL, "category": "printing_office_equipment"})
    [r] = ticket_eval.run_ticket_scenarios([_scenario()], llm_factory=lambda: llm)
    assert r["created"] and r["category_matches"] is False


def test_running_out_of_turns_is_a_recorded_failure():
    llm = StubLLM({"title": None, "description": None, "category": None, "priority": None})
    [r] = ticket_eval.run_ticket_scenarios([_scenario()], llm_factory=lambda: llm, max_turns=2)
    assert not r["created"] and "no ticket after 2 turns" in r["failure"]


def test_ticket_summary_denominators():
    results = [
        {"created": True, "complete": True, "category_matches": True, "turns": 2, "id": "a", "failure": None},
        {"created": True, "complete": False, "category_matches": False, "turns": 4, "id": "b", "failure": None},
        {"created": False, "complete": None, "category_matches": None, "turns": 6, "id": "c", "failure": "x"},
    ]
    s = ticket_eval.ticket_summary(results)
    assert s["creation_success_rate"] == pytest.approx(2 / 3)
    assert s["completeness_rate"] == 0.5 and s["category_agreement"] == 0.5 and s["mean_turns"] == 3
    assert s["failures"] == [{"id": "c", "reason": "x"}]


class SequenceLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, messages, max_tokens=None):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(json.dumps({"category": reply}), "stub", 1, 1, 2, 1.0, "stop")


def test_classification_excludes_llm_errors_and_counts_unclassified_as_wrong():
    examples = [ticket_eval.Example(id=f"c{i}", split="dev", text=f"Employee problem number {i}",
                                    expected_category=cat, validated=True)
                for i, cat in enumerate(["network_wifi", "network_wifi", "account_access", "facilities_maintenance"])]
    llm = SequenceLLM(["network_wifi", "hardware_peripherals", None, LLMError("rate limit")])
    results = ticket_eval.run_classification(examples, llm_factory=lambda: llm)
    summary = ticket_eval.classification_summary(results)
    assert summary["n"] == 3 and summary["n_errors"] == 1 and summary["n_unclassified"] == 1
    assert summary["accuracy"] == pytest.approx(1 / 3)
    assert summary["errors"][0]["id"] == "c3"


def test_dataset_loading_rejects_bad_rows(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"id":"t1","split":"dev","problem":"too short","expected_category":"nope"}\n', encoding="utf-8")
    with pytest.raises(ticket_eval.DatasetError):
        ticket_eval.load_scenarios(path)