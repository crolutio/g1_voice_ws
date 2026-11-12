"""Language model backends: Groq in the cloud, with a local Ollama model as the fallback."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Protocol

import requests

from robot_voice_assistant.ollama_client import ChatMessage
from robot_voice_assistant.token_usage import TokenUsage, groq_usage

__all__ = ["ChatBackend", "FallbackBackend", "GroqClient", "RateLimitedError", "TokenUsage"]

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"


class ChatBackend(Protocol):
    def chat(self, messages: Sequence[ChatMessage], response_schema: dict[str, object]) -> str: ...


class RateLimitedError(RuntimeError):
    """Groq answered 429: too many requests. ``retry_after_s`` says when to try again."""

    def __init__(self, retry_after_s: float) -> None:
        super().__init__(f"rate limited, retry after {retry_after_s:.0f} s")
        self.retry_after_s = retry_after_s


class GroqClient:
    """Groq's OpenAI-compatible chat API in JSON mode (the reply is always a JSON object)."""

    def __init__(self, api_key: str, model: str = DEFAULT_GROQ_MODEL, timeout_s: float = 20.0) -> None:
        if not api_key:
            raise ValueError("A Groq API key is required (set the GROQ_API_KEY environment variable).")
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._model = model
        self._timeout_s = timeout_s
        # GPT-OSS models think before answering; little thinking keeps replies fast, and the thinking
        # tokens count against max_tokens, so they need more room than the spoken answer itself.
        self._reasoning_effort = "low" if model.startswith("openai/gpt-oss") else None
        self.last_usage: TokenUsage | None = None

    def chat(self, messages: Sequence[ChatMessage], response_schema: dict[str, object]) -> str:
        # Groq's JSON mode does not enforce the schema; the response parser checks gesture and motion names.
        response = requests.post(
            GROQ_CHAT_URL,
            json={
                "model": self._model,
                "messages": list(messages),
                "response_format": {"type": "json_object"},
                "temperature": 0.4,
                "max_tokens": 1024,
                **({"reasoning_effort": self._reasoning_effort} if self._reasoning_effort else {}),
            },
            headers=self._headers,
            timeout=self._timeout_s,
        )
        if response.status_code == 429:
            raise RateLimitedError(float(response.headers.get("retry-after", "30")))
        response.raise_for_status()
        body = response.json()
        self.last_usage = groq_usage(body, response.headers)
        content = body["choices"][0]["message"].get("content") or ""
        if not content.strip():
            raise ValueError("Groq returned an empty answer")
        return str(content)


# Without a local model, a rate-limited Groq answer is waited for if Groq says it is this close (it said 4 to 15 s in
# the visitor test; the local model, sharing the GPU with the simulation, took 8 to 60 s).
MAX_WAIT_FOR_PRIMARY_S = 20.0


class FallbackBackend:
    """Use the primary backend; when it fails, answer with the fallback and skip the primary for a while. Without a
    fallback, wait for the primary when it is rate-limited for a short time."""

    def __init__(
        self,
        primary: ChatBackend | None,
        fallback: ChatBackend | None,
        primary_name: str = "groq",
        fallback_name: str = "llama",
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        max_wait_s: float = MAX_WAIT_FOR_PRIMARY_S,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._names = (primary_name, fallback_name)
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._max_wait_s, self._sleep = max_wait_s, sleep
        self._primary_blocked_until = 0.0
        self.last_used = ""
        self.last_error = ""
        self.last_usage: TokenUsage | None = None  # of the model that gave the last answer

    def chat(self, messages: Sequence[ChatMessage], response_schema: dict[str, object]) -> str:
        if self._fallback is None:
            return self._primary_only(messages, response_schema)
        if self._primary is not None and self._clock() >= self._primary_blocked_until:
            try:
                answer = self._primary.chat(messages, response_schema)
                self.last_used, self.last_error = self._names[0], ""
                self.last_usage = getattr(self._primary, "last_usage", None)
                return answer
            except RateLimitedError as error:
                self._block_primary(error.retry_after_s, error)
            except (requests.RequestException, KeyError, IndexError, ValueError) as error:
                self._block_primary(self._cooldown_s, error)
        answer = self._fallback.chat(messages, response_schema)
        self.last_used = self._names[1]
        self.last_usage = getattr(self._fallback, "last_usage", None)
        return answer

    @property
    def has_fallback(self) -> bool:
        return self._fallback is not None

    def _primary_only(self, messages: Sequence[ChatMessage], response_schema: dict[str, object]) -> str:
        """No local model: wait out a short rate limit and ask again; anything else fails."""
        try:
            answer = self._primary.chat(messages, response_schema)
        except RateLimitedError as error:
            if error.retry_after_s > self._max_wait_s:
                raise
            self.last_error = f"rate limited, waited {error.retry_after_s:.0f} s"
            self._sleep(error.retry_after_s)
            answer = self._primary.chat(messages, response_schema)
        else:
            self.last_error = ""
        self.last_used = self._names[0]
        self.last_usage = getattr(self._primary, "last_usage", None)
        return answer

    def _block_primary(self, seconds: float, error: Exception) -> None:
        self._primary_blocked_until = self._clock() + seconds
        self.last_error = f"{type(error).__name__}: {error}"
