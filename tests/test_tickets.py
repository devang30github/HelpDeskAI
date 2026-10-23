def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["database"] == "ok"
    assert body["index"]["built"] is False  # no index in the test environment


def test_create_ticket_has_required_fields(client, employee_headers, ticket_payload):
    r = client.post("/api/tickets", json=ticket_payload, headers=employee_headers)
    assert r.status_code == 201
    body = r.json()
    assert body["ticket_id"].startswith("HD-")
    assert body["status"] == "open"
    assert body["assignee"] is None
    assert body["requester"] == "alice"
    assert body["created_at"] and body["updated_at"] and body["due_at"]


def test_ticket_creation_requires_login(client, ticket_payload):
    assert client.post("/api/tickets", json=ticket_payload).status_code == 401


def test_requester_cannot_be_spoofed(client, employee_headers, ticket_payload):
    r = client.post("/api/tickets", json={**ticket_payload, "requester": "bob"}, headers=employee_headers)
    assert r.status_code == 201
    assert r.json()["requester"] == "alice"


def test_invalid_category_rejected(client, employee_headers, ticket_payload):
    ticket_payload["category"] = "coffee_machine"
    assert client.post("/api/tickets", json=ticket_payload, headers=employee_headers).status_code == 422


def test_idempotency_key_prevents_duplicates(client, employee_headers, ticket_payload, admin):
    ticket_payload["idempotency_key"] = "conv-1-ticket"
    first = client.post("/api/tickets", json=ticket_payload, headers=employee_headers)
    second = client.post("/api/tickets", json=ticket_payload, headers=employee_headers)
    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json()["ticket_id"] == second.json()["ticket_id"]
    assert len(client.get("/api/tickets", headers=admin).json()) == 1


def test_idempotency_keys_are_scoped_per_user(client, auth_for, ticket_payload):
    ticket_payload["idempotency_key"] = "same-key"
    a = client.post("/api/tickets", json=ticket_payload, headers=auth_for("alice"))
    b = client.post("/api/tickets", json=ticket_payload, headers=auth_for("bob"))
    assert a.status_code == 201 and b.status_code == 201
    assert a.json()["ticket_id"] != b.json()["ticket_id"]
    assert b.json()["requester"] == "bob"


def test_admin_endpoints_require_key(client, employee_headers, ticket_payload):
    client.post("/api/tickets", json=ticket_payload, headers=employee_headers)
    assert client.get("/api/tickets").status_code == 401
    assert client.get("/api/tickets", headers={"X-Admin-Key": "wrong"}).status_code == 401
    # an employee token must not grant admin access
    assert client.get("/api/tickets", headers=employee_headers).status_code == 401


def test_filter_and_search(client, employee_headers, ticket_payload, admin):
    client.post("/api/tickets", json=ticket_payload, headers=employee_headers)
    other = {**ticket_payload, "title": "Printer on floor 3 is jammed",
             "category": "printing_office_equipment", "priority": "low"}
    client.post("/api/tickets", json=other, headers=employee_headers)

    r = client.get("/api/tickets", params={"category": "printing_office_equipment"}, headers=admin)
    assert [t["title"] for t in r.json()] == ["Printer on floor 3 is jammed"]
    r = client.get("/api/tickets", params={"q": "wi-fi"}, headers=admin)
    assert len(r.json()) == 1
    r = client.get("/api/tickets", params={"status": "bogus"}, headers=admin)
    assert r.status_code == 422


def test_lifecycle_and_assignment(client, employee_headers, ticket_payload, admin):
    tid = client.post("/api/tickets", json=ticket_payload, headers=employee_headers).json()["ticket_id"]

    r = client.patch(f"/api/tickets/{tid}", json={"status": "in_progress", "assignee": "Asha Verma"}, headers=admin)
    assert r.status_code == 200
    assert r.json()["status"] == "in_progress" and r.json()["assignee"] == "Asha Verma"

    # cannot resolve without notes
    r = client.patch(f"/api/tickets/{tid}", json={"status": "resolved"}, headers=admin)
    assert r.status_code == 422

    r = client.patch(f"/api/tickets/{tid}", json={"status": "resolved", "resolution_notes": "Replaced access point."}, headers=admin)
    assert r.status_code == 200

    r = client.patch(f"/api/tickets/{tid}", json={"status": "closed"}, headers=admin)
    assert r.status_code == 200
    # closed is terminal
    r = client.patch(f"/api/tickets/{tid}", json={"status": "open"}, headers=admin)
    assert r.status_code == 409


def test_invalid_transition_and_assignee(client, employee_headers, ticket_payload, admin):
    tid = client.post("/api/tickets", json=ticket_payload, headers=employee_headers).json()["ticket_id"]
    assert client.patch(f"/api/tickets/{tid}", json={"status": "resolved"}, headers=admin).status_code in (409, 422)
    assert client.patch(f"/api/tickets/{tid}", json={"assignee": "Nobody"}, headers=admin).status_code == 422
    assert client.patch("/api/tickets/HD-99999", json={"priority": "low"}, headers=admin).status_code == 404