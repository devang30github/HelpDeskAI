"""LLM access through Groq's OpenAI-compatible endpoint, using the openai client.
Several API keys are used round-robin; a rate-limited key is benched and the next one is tried."""
import logging
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Protocol
import re
import openai
from openai import OpenAI

from app.core.config import Settings, get_settings
from app.core.keypool import KeyPool

logger = logging.getLogger(__name__)

DEFAULT_COOLDOWN_SECONDS = 20.0


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


def _retry_after_seconds(error: Exception) -> float:
    try:
        value = error.response.headers.get("retry-after")
        return min(max(float(value), 1.0), 120.0) if value else DEFAULT_COOLDOWN_SECONDS
    except Exception:
        return DEFAULT_COOLDOWN_SECONDS


class GroqLLM:
    def __init__(
        self,
        settings: Settings | None = None,
        client: OpenAI | None = None,
        pool: KeyPool | None = None,
        client_factory: Callable[[str], object] | None = None,
    ):
        self.settings = settings or get_settings()
        self.model = self.settings.groq_model
        self.pool = pool or KeyPool(self.settings.groq_key_list())
        if len(self.pool) == 0:
            raise LLMConfigError("GROQ_API_KEY (or GROQ_API_KEYS) is not set (add it to your .env file)")
        self._injected_client = client          # tests: one fake client for every key
        self._client_factory = client_factory   # tests: a fake client per key
        self._clients: dict[str, object] = {}
        self._lock = threading.Lock()

    def config(self) -> dict:
        s = self.settings
        return {
            "model": s.groq_model,
            "temperature": s.llm_temperature,
            "max_tokens": s.llm_max_tokens,
            "reasoning_effort": s.groq_reasoning_effort or None,
            "api_keys": len(self.pool),
        }

    def _client_for(self, key: str):
        if self._injected_client is not None:
            return self._injected_client
        with self._lock:
            if key not in self._clients:
                if self._client_factory is not None:
                    self._clients[key] = self._client_factory(key)
                else:
                    s = self.settings
                    self._clients[key] = OpenAI(
                        api_key=key,
                        base_url=s.groq_base_url,
                        timeout=s.groq_timeout_seconds,
                        # with several keys we fail over ourselves instead of retrying the same key
                        max_retries=s.groq_max_retries if len(self.pool) == 1 else 0,
                    )
            return self._clients[key]

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

        single = len(self.pool) == 1
        attempts = 1 if single else len(self.pool) + 1  # one try per key, plus one after the earliest cooldown
        start = time.perf_counter()
        response = None
        for _ in range(attempts):
            try:
                key, wait = self.pool.pick()
            except RuntimeError:
                raise LLMError("no usable Groq API keys (all were rejected as invalid)")
            if wait > 0:
                logger.info("all API keys are cooling down; waiting %.1fs", wait)
                time.sleep(wait)
            label = self.pool.label(key)
            try:
                response = self._client_for(key).chat.completions.create(**kwargs)
                break
            except openai.APITimeoutError:
                raise LLMError(f"LLM request timed out after {s.groq_timeout_seconds}s")
            except openai.APIConnectionError:
                raise LLMError("Could not connect to the LLM provider")
            except openai.RateLimitError as e:
                if single:
                    raise LLMError("LLM provider rate limit reached; try again shortly")
                cooldown = _retry_after_seconds(e)
                self.pool.cooldown(key, cooldown)
                logger.warning("%s was rate limited; benched for %.0fs", label, cooldown)
            except openai.AuthenticationError:
                if single:
                    raise LLMError("LLM provider rejected the API key (HTTP 401)")
                self.pool.disable(key)
                logger.error("%s was rejected as invalid and is disabled", label)
            except openai.APIStatusError as e:
                # Never include request bodies or keys in error text.
                raise LLMError(f"LLM provider returned HTTP {e.status_code}")
        else:
            raise LLMError("The LLM request failed on every API key (rate limited or rejected); try again shortly")
        latency_ms = (time.perf_counter() - start) * 1000  # includes any failover waiting

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
        logger.info("llm call model=%s key=%s latency_ms=%.0f tokens=%d finish=%s",
                    result.model, label, latency_ms, result.total_tokens, finish)
        return result


@lru_cache
def get_llm() -> LLMClient:
    return GroqLLM()