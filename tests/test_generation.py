from types import SimpleNamespace

import httpx
import openai
import pytest

from app.core.config import Settings
from app.rag.generation import GroqLLM, LLMConfigError, LLMError


def _response(content="Hello [1]", finish="stop", usage=(10, 5, 15)):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1], total_tokens=usage[2]),
    )


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _settings(**overrides):
    return Settings(groq_api_key="test-key", groq_model="test-model", **overrides)


def test_successful_call_parses_text_and_usage():
    client = FakeClient(_response())
    result = GroqLLM(_settings(), client=client).complete([{"role": "user", "content": "hi"}])
    assert result.text == "Hello [1]" and result.total_tokens == 15 and result.model == "test-model"
    call = client.calls[0]
    assert call["model"] == "test-model" and call["max_completion_tokens"] == _settings().llm_max_tokens
    assert "extra_body" not in call


def test_reasoning_effort_is_sent_when_configured():
    client = FakeClient(_response())
    GroqLLM(_settings(groq_reasoning_effort="low"), client=client).complete([])
    assert client.calls[0]["extra_body"] == {"reasoning_effort": "low"}


def test_empty_answer_from_truncation_gives_helpful_error():
    client = FakeClient(_response(content="", finish="length"))
    with pytest.raises(LLMError, match="LLM_MAX_TOKENS"):
        GroqLLM(_settings(), client=client).complete([])


def test_missing_api_key_raises_config_error():
    with pytest.raises(LLMConfigError):
        GroqLLM(Settings(groq_api_key=""))


def test_timeout_is_mapped_to_llm_error():
    err = openai.APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
    with pytest.raises(LLMError, match="timed out"):
        GroqLLM(_settings(), client=FakeClient(error=err)).complete([])