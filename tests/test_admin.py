from datetime import timedelta

from app.core.timeutils import utcnow
from app.database.session import SessionLocal
from app.models.conversation import Conversation, Message
from app.models.ticket import Ticket

import json


def _make_ticket(client, headers, **overrides):
    payload = {"title": "Wi-Fi drops often", "description": "Laptop loses Wi-Fi every five minutes.",
               "category": "network_wifi", "priority": "high", **overrides}
    r = client.post("/api/tickets", json=payload, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()["ticket_id"]


def test_admin_routes_require_the_key(client, employee_headers):
    assert client.get("/api/admin/stats").status_code == 401
    assert client.get("/api/admin/stats", headers=employee_headers).status_code == 401  # employee token is not admin


def test_stats_counts_are_zero_filled_and_correct(client, employee_headers, admin):
    a = _make_ticket(client, employee_headers)
    _make_ticket(client, employee_headers, title="Printer jam floor 3",
                 category="printing_office_equipment", priority="low")
    client.patch(f"/api/tickets/{a}", json={"status": "in_progress", "assignee": "Asha Verma"}, headers=admin)

    s = client.get("/api/admin/stats", headers=admin).json()
    assert s["total"] == 2
    assert s["by_status"] == {"open": 1, "in_progress": 1, "resolved": 0, "closed": 0}
    assert s["by_priority"]["high"] == 1 and s["by_priority"]["critical"] == 0
    assert s["by_category"]["printing_office_equipment"] == 1
    assert s["unassigned_open"] == 1 and s["overdue"] == 0


def test_overdue_flag_and_count(client, employee_headers, admin):
    tid = _make_ticket(client, employee_headers)
    with SessionLocal() as db:
        ticket = db.query(Ticket).filter_by(ticket_id=tid).one()
        ticket.due_at = utcnow() - timedelta(hours=1)
        db.commit()

    assert client.get("/api/tickets", headers=admin).json()[0]["is_overdue"] is True
    assert client.get("/api/admin/stats", headers=admin).json()["overdue"] == 1

    client.patch(f"/api/tickets/{tid}", json={"status": "in_progress"}, headers=admin)
    client.patch(f"/api/tickets/{tid}", json={"status": "resolved", "resolution_notes": "Fixed."}, headers=admin)
    assert client.get("/api/tickets", headers=admin).json()[0]["is_overdue"] is False  # resolved never overdue
    assert client.get("/api/admin/stats", headers=admin).json()["overdue"] == 0


def test_unassigned_filter(client, employee_headers, admin):
    a = _make_ticket(client, employee_headers)
    b = _make_ticket(client, employee_headers, title="Second ticket here")
    client.patch(f"/api/tickets/{a}", json={"assignee": "Asha Verma"}, headers=admin)
    r = client.get("/api/tickets", params={"assignee": "unassigned"}, headers=admin).json()
    assert [t["ticket_id"] for t in r] == [b]


def test_meta_exposes_status_transitions(client):
    transitions = client.get("/api/meta").json()["status_transitions"]
    assert transitions["open"] == ["closed", "in_progress"]
    assert transitions["closed"] == []


def test_ticket_conversation_endpoint(client, employee_headers, admin):
    with SessionLocal() as db:
        db.add(Conversation(id="conv-123", username="alice", state_json="{}"))
        db.add(Message(conversation_id="conv-123", role="user", content="My VPN times out"))
        db.add(Message(
            conversation_id="conv-123", role="assistant", content="Restart the client [1].",
            meta_json=json.dumps({"intent": "kb_question",
                                  "sources": [{"source_name": "vpn-guide.pdf", "cited": True},
                                              {"source_name": "other.txt", "cited": False}]}),
        ))
        db.commit()
    tid = _make_ticket(client, employee_headers, conversation_id="conv-123")

    r = client.get(f"/api/admin/tickets/{tid}/conversation", headers=admin)
    assert r.status_code == 200
    msgs = r.json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["meta"] == {"intent": "kb_question", "sources": ["vpn-guide.pdf"]}


def test_ticket_conversation_edge_cases(client, employee_headers, admin):
    tid = _make_ticket(client, employee_headers)  # no conversation linked
    assert client.get(f"/api/admin/tickets/{tid}/conversation", headers=admin).json() == []
    assert client.get("/api/admin/tickets/HD-99999/conversation", headers=admin).status_code == 404
    assert client.get(f"/api/admin/tickets/{tid}/conversation").status_code == 401