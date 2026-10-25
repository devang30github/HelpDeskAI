from types import SimpleNamespace

import httpx
import openai
import pytest

from app.core.config import Settings
from app.rag import generation
from app.rag.generation import GroqLLM, LLMError


def _response():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok [1]"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _http(status, headers=None):
    return httpx.Response(status, headers=headers or {}, request=httpx.Request("POST", "https://example.invalid"))


def _rate_limited(retry_after="3"):
    return openai.RateLimitError("limited", response=_http(429, {"retry-after": retry_after}), body=None)


def _invalid():
    return openai.AuthenticationError("bad key", response=_http(401), body=None)


class Factory:
    """One fake client per key. A behaviour is either an exception to raise or a response to return."""

    def __init__(self, behaviours):
        self.behaviours, self.calls = behaviours, []

    def __call__(self, key):
        def create(**kwargs):
            self.calls.append(key)
            behaviour = self.behaviours[key]
            if isinstance(behaviour, Exception):
                raise behaviour
            return behaviour

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _settings(keys):
    return Settings(groq_api_key="", groq_api_keys=keys, groq_model="test-model")


def test_calls_rotate_round_robin():
    factory = Factory({k: _response() for k in ("k1", "k2", "k3")})
    llm = GroqLLM(_settings("k1,k2,k3"), client_factory=factory)
    for _ in range(6):
        llm.complete([])
    assert factory.calls == ["k1", "k2", "k3", "k1", "k2", "k3"]


def test_rate_limited_key_is_benched_and_the_next_key_is_used():
    factory = Factory({"k1": _rate_limited(), "k2": _response()})
    llm = GroqLLM(_settings("k1,k2"), client_factory=factory)
    assert llm.complete([]).text == "ok [1]"
    llm.complete([])
    assert factory.calls == ["k1", "k2", "k2"]  # k1 stays benched for its retry-after period


def test_invalid_key_is_disabled_and_the_others_keep_working():
    factory = Factory({"k1": _invalid(), "k2": _response()})
    llm = GroqLLM(_settings("k1,k2"), client_factory=factory)
    llm.complete([])
    llm.complete([])
    assert factory.calls == ["k1", "k2", "k2"]


def test_all_keys_limited_waits_for_the_earliest_then_fails(monkeypatch):
    sleeps = []
    monkeypatch.setattr(generation.time, "sleep", lambda s: sleeps.append(s))
    factory = Factory({"k1": _rate_limited(), "k2": _rate_limited()})
    llm = GroqLLM(_settings("k1,k2"), client_factory=factory)
    with pytest.raises(LLMError, match="every API key"):
        llm.complete([])
    assert factory.calls == ["k1", "k2", "k1"] and len(sleeps) == 1 and 0 < sleeps[0] <= 3


def test_single_key_keeps_the_old_behaviour():
    factory = Factory({"k1": _rate_limited()})
    llm = GroqLLM(_settings("k1"), client_factory=factory)
    with pytest.raises(LLMError, match="rate limit reached"):
        llm.complete([])
    assert factory.calls == ["k1"]