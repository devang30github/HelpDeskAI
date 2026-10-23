import json
from types import SimpleNamespace

import pytest
from fastapi import Depends
from sqlalchemy.orm import Session

from app.api.chat import get_chat_service
from app.api.deps import get_db
from app.main import app
from app.rag.generation import LLMError, LLMResult
from app.rag.pipeline import RAGResult, SourceRef
from app.services.chat_service import ChatService


class FakePipeline:
    def __init__(self, env):
        self.env = env

    def answer(self, question, history=None):
        self.env.pipeline_calls.append((question, history))
        if self.env.pipeline_error:
            raise LLMError("boom")
        sources = [SourceRef(1, "vpn-guide::chunk-000", "vpn-guide", "vpn-guide.pdf", "VPN Guide", 0.8, 1,
                             self.env.answerable)]
        return RAGResult(
            question=question, answer=self.env.answer, answerable=self.env.answerable, sources=sources,
            retrieved=[], timings={"retrieval_ms": 1.0, "generation_ms": 2.0, "total_ms": 3.0},
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            config={"pipeline": "baseline_rag_v1", "prompt_version": "v1", "retriever": "fake", "top_k": 4},
        )


class ScriptedLLM:
    def __init__(self, env):
        self.env = env

    def complete(self, messages, max_tokens=None):
        system = messages[0]["content"].lower()
        if "intent classifier" in system:
            text = json.dumps({"intent": self.env.intent})
        elif "ticket drafting" in system:
            text = json.dumps(self.env.draft)
        else:
            text = "unexpected prompt"
        return LLMResult(text, "fake", 1, 1, 2, 1.0, "stop")


EMPTY_DRAFT = {"title": None, "description": None, "category": None, "priority": None}


@pytest.fixture
def env(client):
    env = SimpleNamespace(intent="kb_question", draft=dict(EMPTY_DRAFT), answer="Restart the client [1].",
                          answerable=True, pipeline_error=False, pipeline_calls=[])
    pipeline, llm = FakePipeline(env), ScriptedLLM(env)

    def override(db: Session = Depends(get_db)):
        return ChatService(db, pipeline_factory=lambda: pipeline, llm_factory=lambda: llm)

    app.dependency_overrides[get_chat_service] = override
    yield env
    app.dependency_overrides.pop(get_chat_service, None)


def say(client, headers, message, conversation_id=None):
    body = {"message": message}
    if conversation_id:
        body["conversation_id"] = conversation_id
    return client.post("/api/chat", json=body, headers=headers)


def test_kb_question_returns_cited_answer_and_persists(client, env, employee_headers):
    r = say(client, employee_headers, "My VPN says VPN-612")
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "kb_question" and body["answerable"] is True and body["mode"] == "idle"
    assert body["reply"] == "Restart the client [1]."
    assert body["sources"][0]["source_name"] == "vpn-guide.pdf" and body["sources"][0]["cited"] is True
    assert body["debug"]["timings"]["total_ms"] == 3.0

    cid = body["conversation_id"]
    say(client, employee_headers, "And what if it still fails?", cid)
    history = env.pipeline_calls[1][1]
    assert [m["role"] for m in history] == ["user", "assistant"]

    conv = client.get(f"/api/chat/{cid}", headers=employee_headers).json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant", "user", "assistant"]


def test_out_of_scope_is_rejected_without_retrieval(client, env, employee_headers):
    env.intent = "out_of_scope"
    body = say(client, employee_headers, "How many leave days do I have?").json()
    assert body["intent"] == "out_of_scope" and "outside" in body["reply"]
    assert env.pipeline_calls == []


def test_unanswerable_question_offers_a_ticket_and_flow_creates_it(client, env, employee_headers, admin):
    env.answerable, env.answer = False, "The documents do not cover badge readers."
    first = say(client, employee_headers, "Why is my badge reader beeping?").json()
    assert first["answerable"] is False and first["mode"] == "ticket_offered"
    assert "ticket" in first["reply"] and first["quick_replies"]
    cid = first["conversation_id"]

    env.draft = {"title": "Badge reader beeping", "description": "The badge reader at the main door beeps constantly.",
                 "category": "facilities_maintenance", "priority": "low"}
    summary = say(client, employee_headers, "Yes, create a ticket", cid).json()
    assert summary["mode"] == "confirming"
    assert "Badge reader beeping" in summary["reply"] and "Facilities and maintenance" in summary["reply"]
    assert summary["quick_replies"][0].startswith("Yes")

    done = say(client, employee_headers, "Yes, create the ticket", cid).json()
    assert done["mode"] == "idle" and done["ticket"]["ticket_id"].startswith("HD-")

    tickets = client.get("/api/tickets", headers=admin).json()
    assert len(tickets) == 1
    t = tickets[0]
    assert t["requester"] == "alice" and t["conversation_id"] == cid and t["category"] == "facilities_maintenance"
    assert "badge reader" in t["transcript_summary"].lower()


def test_explicit_request_collects_missing_fields(client, env, employee_headers, admin):
    first = say(client, employee_headers, "Please create a ticket").json()
    assert first["mode"] == "collecting" and "describe" in first["reply"]
    cid = first["conversation_id"]

    env.draft = {**EMPTY_DRAFT, "priority": "medium"}  # model suggests a priority but no category
    second = say(client, employee_headers, "My monitor flickers whenever I plug in the dock.", cid).json()
    assert second["mode"] == "collecting" and "Hardware and peripherals" in second["quick_replies"]

    third = say(client, employee_headers, "Hardware and peripherals", cid).json()
    assert third["mode"] == "confirming" and "Title: My monitor flickers" in third["reply"]

    done = say(client, employee_headers, "yes", cid).json()
    assert done["ticket"]["category"] == "hardware_peripherals" and done["ticket"]["priority"] == "medium"
    assert len(client.get("/api/tickets", headers=admin).json()) == 1


def test_short_description_is_re_requested(client, env, employee_headers):
    cid = say(client, employee_headers, "Please create a ticket").json()["conversation_id"]
    r = say(client, employee_headers, "help", cid).json()
    assert r["mode"] == "collecting" and "more detail" in r["reply"]


def test_cancelling_creates_no_ticket(client, env, employee_headers, admin):
    env.draft = {"title": "Printer jam on floor 3", "description": "The floor 3 printer has a paper jam.",
                 "category": "printing_office_equipment", "priority": "low"}
    first = say(client, employee_headers, "Please create a ticket").json()
    assert first["mode"] == "confirming"
    r = say(client, employee_headers, "cancel", first["conversation_id"]).json()
    assert r["mode"] == "idle"
    assert client.get("/api/tickets", headers=admin).json() == []


def test_conversations_belong_to_their_owner(client, env, auth_for):
    alice, bob = auth_for("alice"), auth_for("bob")
    cid = say(client, alice, "My VPN says VPN-612").json()["conversation_id"]
    assert client.get(f"/api/chat/{cid}", headers=bob).status_code == 404
    assert say(client, bob, "hello again there", cid).status_code == 404


def test_requires_login_and_validates_input(client, env, employee_headers):
    assert client.post("/api/chat", json={"message": "hi there"}).status_code == 401
    assert say(client, employee_headers, "").status_code == 422
    assert say(client, employee_headers, "x" * 2001).status_code == 422


def test_llm_failure_returns_503_and_stores_nothing(client, env, employee_headers):
    env.pipeline_error = True
    r = say(client, employee_headers, "My VPN says VPN-612")
    assert r.status_code == 503 and "temporarily unavailable" in r.json()["detail"]