"""LLM access through Groq's OpenAI-compatible endpoint, using the openai client."""
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import openai
from openai import OpenAI

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class LLMConfigError(LLMError):
    pass


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    finish_reason: str | None


class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[dict], max_tokens: int | None = None) -> LLMResult: ...
    def config(self) -> dict: ...


class GroqLLM:
    def __init__(self, settings: Settings | None = None, client: OpenAI | None = None):
        self.settings = settings or get_settings()
        self.model = self.settings.groq_model
        if client is None:
            if not self.settings.groq_api_key:
                raise LLMConfigError("GROQ_API_KEY is not set (add it to your .env file)")
            client = OpenAI(
                api_key=self.settings.groq_api_key,
                base_url=self.settings.groq_base_url,
                timeout=self.settings.groq_timeout_seconds,
                max_retries=self.settings.groq_max_retries,
            )
        self.client = client

    def config(self) -> dict:
        s = self.settings
        return {
            "model": s.groq_model,
            "temperature": s.llm_temperature,
            "max_tokens": s.llm_max_tokens,
            "reasoning_effort": s.groq_reasoning_effort or None,
        }

    def complete(self, messages: list[dict], max_tokens: int | None = None) -> LLMResult:
        s = self.settings
        kwargs: dict = {
            "model": s.groq_model,
            "messages": messages,
            "temperature": s.llm_temperature,
            "max_completion_tokens": max_tokens or s.llm_max_tokens,
        }
        if s.groq_reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": s.groq_reasoning_effort}

        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**kwargs)
        except openai.APITimeoutError:
            raise LLMError(f"LLM request timed out after {s.groq_timeout_seconds}s")
        except openai.APIConnectionError:
            raise LLMError("Could not connect to the LLM provider")
        except openai.RateLimitError:
            raise LLMError("LLM provider rate limit reached; try again shortly")
        except openai.APIStatusError as e:
            # Never include request bodies or keys in error text.
            raise LLMError(f"LLM provider returned HTTP {e.status_code}")
        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        finish = getattr(choice, "finish_reason", None)
        if not text:
            hint = " Increase LLM_MAX_TOKENS (reasoning models spend tokens thinking)." if finish == "length" else ""
            raise LLMError(f"LLM returned an empty answer (finish_reason={finish}).{hint}")

        usage = getattr(response, "usage", None)
        result = LLMResult(
            text=text,
            model=s.groq_model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            latency_ms=latency_ms,
            finish_reason=finish,
        )
        logger.info(
            "llm call model=%s latency_ms=%.0f tokens=%d finish=%s",
            result.model, latency_ms, result.total_tokens, finish,
        )
        return result


@lru_cache
def get_llm() -> LLMClient:
    return GroqLLM()