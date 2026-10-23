import pytest

from app.rag.generation import LLMConfigError, LLMError, LLMResult
from app.services.intent import classify_intent


class StubLLM:
    def __init__(self, text=None, error=None):
        self.text, self.error, self.calls = text, error, 0

    def complete(self, messages, max_tokens=None):
        self.calls += 1
        if self.error:
            raise self.error
        return LLMResult(self.text, "stub", 1, 1, 2, 1.0, "stop")


def test_greeting_rule_skips_the_llm():
    llm = StubLLM('{"intent": "kb_question"}')
    assert classify_intent("hello", [], lambda: llm) == ("chitchat", "rule")
    assert llm.calls == 0


def test_explicit_ticket_rule():
    llm = StubLLM()
    assert classify_intent("Please create a ticket for this", [], lambda: llm) == ("ticket_request", "rule")
    assert llm.calls == 0


def test_how_to_ticket_question_goes_to_the_llm():
    llm = StubLLM('{"intent": "kb_question"}')
    assert classify_intent("How do I create a ticket?", [], lambda: llm) == ("kb_question", "llm")
    assert llm.calls == 1


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"intent": "out_of_scope"}', ("out_of_scope", "llm")),
        ('Sure! ```json\n{"intent": "chitchat"}\n```', ("chitchat", "llm")),
        ('{"intent": "made_up_label"}', ("kb_question", "fallback")),
        ("no json here", ("kb_question", "fallback")),
    ],
)
def test_llm_classification_is_validated(reply, expected):
    assert classify_intent("what about leave days", [], lambda: StubLLM(reply)) == expected


def test_failures_fall_back_to_kb_question():
    assert classify_intent("my wifi is down", [], lambda: StubLLM(error=LLMError("boom"))) == ("kb_question", "fallback")

    def no_key():
        raise LLMConfigError("no key")

    assert classify_intent("my wifi is down", [], no_key) == ("kb_question", "fallback")