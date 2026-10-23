from app.rag.generation import LLMError, LLMResult
from app.services.ticket_drafting import sanitize_draft, suggest_draft


class StubLLM:
    def __init__(self, text=None, error=None):
        self.text, self.error = text, error

    def complete(self, messages, max_tokens=None):
        if self.error:
            raise self.error
        return LLMResult(self.text, "stub", 1, 1, 2, 1.0, "stop")


def test_sanitize_validates_every_field():
    d = sanitize_draft({"title": "  VPN   fails  ", "description": "VPN times out at home.",
                        "category": "Network_WiFi", "priority": "URGENT"})
    assert d.title == "VPN fails" and d.category == "network_wifi" and d.priority is None


def test_sanitize_rejects_short_or_wrong_types():
    d = sanitize_draft({"title": "ab", "description": "short", "category": 5, "priority": None})
    assert d.to_dict() == {"title": None, "description": None, "category": None, "priority": None}
    assert sanitize_draft(None).description is None


def test_suggest_draft_parses_llm_json():
    reply = '{"title": "Badge reader beeps", "description": "The badge reader at the door beeps constantly.", "category": "facilities_maintenance", "priority": "low"}'
    draft = suggest_draft("Employee: badge reader beeps", lambda: StubLLM(reply))
    assert draft.category == "facilities_maintenance" and draft.priority == "low"


def test_suggest_draft_never_raises():
    assert suggest_draft("x", lambda: StubLLM(error=LLMError("down"))).to_dict()["description"] is None
    assert suggest_draft("x", lambda: StubLLM("not json")).to_dict()["category"] is None