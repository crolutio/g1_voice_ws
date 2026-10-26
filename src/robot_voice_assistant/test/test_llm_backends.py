from collections.abc import Sequence

import pytest
import requests

from robot_voice_assistant.llm_backends import FallbackBackend, GroqClient, RateLimitedError
from robot_voice_assistant.ollama_client import ChatMessage

MESSAGES: list[ChatMessage] = [{"role": "system", "content": "Answer in JSON."}, {"role": "user", "content": "Hi"}]
SCHEMA: dict[str, object] = {"type": "object"}


class FakeResponse:
    def __init__(self, status: int, body: dict[str, object], headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def json(self) -> dict[str, object]:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _ok(content: str) -> FakeResponse:
    return FakeResponse(200, {"choices": [{"message": {"content": content}}]})


def test_groq_sends_openai_style_request(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_post(url: str, json: dict[str, object], headers: dict[str, str], timeout: float) -> FakeResponse:
        seen.update(url=url, json=json, headers=headers)
        return _ok('{"reply": "Hello", "gesture": "none"}')

    monkeypatch.setattr(requests, "post", fake_post)
    client = GroqClient(api_key="test-key", model="llama-3.3-70b-versatile")
    assert client.chat(MESSAGES, SCHEMA) == '{"reply": "Hello", "gesture": "none"}'
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    payload = seen["json"]
    assert payload["model"] == "llama-3.3-70b-versatile"
    assert payload["messages"] == MESSAGES
    assert payload["response_format"] == {"type": "json_object"}


def test_groq_rate_limit_reports_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(429, {}, {"retry-after": "12"}))
    with pytest.raises(RateLimitedError) as error:
        GroqClient(api_key="k", model="m").chat(MESSAGES, SCHEMA)
    assert error.value.retry_after_s == 12.0


def test_groq_requires_a_key() -> None:
    with pytest.raises(ValueError):
        GroqClient(api_key="", model="m")


class FakeBackend:
    def __init__(self, name: str, outcomes: list[object]) -> None:
        self.name = name
        self.outcomes = list(outcomes)
        self.calls = 0

    def chat(self, messages: Sequence[ChatMessage], response_schema: dict[str, object]) -> str:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return str(outcome)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_primary_answers_when_it_works() -> None:
    primary, fallback = FakeBackend("groq", ["from groq"]), FakeBackend("llama", ["from llama"])
    backend = FallbackBackend(primary, fallback)
    assert backend.chat(MESSAGES, SCHEMA) == "from groq"
    assert backend.last_used == "groq"
    assert fallback.calls == 0


def test_fallback_answers_when_primary_fails() -> None:
    primary = FakeBackend("groq", [requests.ConnectionError("offline")])
    fallback = FakeBackend("llama", ["from llama"])
    backend = FallbackBackend(primary, fallback)
    assert backend.chat(MESSAGES, SCHEMA) == "from llama"
    assert backend.last_used == "llama"
    assert "offline" in backend.last_error


def test_primary_is_skipped_during_cooldown_then_retried() -> None:
    clock = FakeClock()
    primary = FakeBackend("groq", [requests.Timeout("slow"), "groq is back"])
    fallback = FakeBackend("llama", ["llama 1", "llama 2"])
    backend = FallbackBackend(primary, fallback, cooldown_s=30.0, clock=clock)
    backend.chat(MESSAGES, SCHEMA)
    clock.now = 10.0
    assert backend.chat(MESSAGES, SCHEMA) == "llama 2"
    assert primary.calls == 1  # not retried during the cooldown
    clock.now = 31.0
    assert backend.chat(MESSAGES, SCHEMA) == "groq is back"


def test_rate_limit_cooldown_follows_retry_after() -> None:
    clock = FakeClock()
    primary = FakeBackend("groq", [RateLimitedError(120.0), "ok"])
    fallback = FakeBackend("llama", ["a", "b"])
    backend = FallbackBackend(primary, fallback, cooldown_s=30.0, clock=clock)
    backend.chat(MESSAGES, SCHEMA)
    clock.now = 60.0
    backend.chat(MESSAGES, SCHEMA)
    assert primary.calls == 1  # still rate limited after 60 s
    clock.now = 121.0
    assert backend.chat(MESSAGES, SCHEMA) == "ok"


def test_error_when_both_backends_fail() -> None:
    primary = FakeBackend("groq", [requests.ConnectionError("offline")])
    fallback = FakeBackend("llama", [requests.ConnectionError("ollama down")])
    with pytest.raises(requests.ConnectionError):
        FallbackBackend(primary, fallback).chat(MESSAGES, SCHEMA)


def test_no_primary_means_fallback_only() -> None:
    fallback = FakeBackend("llama", ["only llama"])
    backend = FallbackBackend(None, fallback)
    assert backend.chat(MESSAGES, SCHEMA) == "only llama"
