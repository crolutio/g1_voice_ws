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
    client = GroqClient(api_key="test-key", model="qwen/qwen3.8-27b")
    assert client.chat(MESSAGES, SCHEMA) == '{"reply": "Hello", "gesture": "none"}'
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    payload = seen["json"]
    assert payload["model"] == "qwen/qwen3.8-27b"
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


def test_reasoning_models_get_low_effort_and_room_to_think(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, object]] = []

    def fake_post(url: str, json: dict[str, object], headers: dict[str, str], timeout: float) -> FakeResponse:
        seen.append(json)
        return _ok('{"reply": "Hi", "gesture": "none"}')

    monkeypatch.setattr(requests, "post", fake_post)
    GroqClient(api_key="k", model="openai/gpt-oss-120b").chat(MESSAGES, SCHEMA)
    GroqClient(api_key="k", model="qwen/qwen3.8-27b").chat(MESSAGES, SCHEMA)
    assert seen[0]["reasoning_effort"] == "low"
    assert seen[0]["max_tokens"] >= 1024  # thinking tokens count against the limit
    assert "reasoning_effort" not in seen[1]


def test_empty_answer_counts_as_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # A reasoning model that runs out of tokens returns no content; the fallback should answer instead.
    monkeypatch.setattr(requests, "post", lambda *a, **k: _ok(""))
    with pytest.raises(ValueError):
        GroqClient(api_key="k", model="openai/gpt-oss-120b").chat(MESSAGES, SCHEMA)


def test_groq_reports_the_tokens_used_and_what_is_left_of_the_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"choices": [{"message": {"content": '{"reply": "Hi"}'}}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 25, "total_tokens": 1225}}
    headers = {"x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "998",
               "x-ratelimit-limit-tokens": "6000", "x-ratelimit-remaining-tokens": "4775"}
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(200, body, headers))
    client = GroqClient(api_key="k", model="m")
    client.chat(MESSAGES, SCHEMA)
    usage = client.last_usage
    assert (usage.prompt_tokens, usage.completion_tokens) == (1200, 25)
    assert (usage.requests_per_day, usage.requests_left_today) == (1000, 998)
    assert (usage.tokens_per_minute, usage.tokens_left_this_minute) == (6000, 4775)


def test_groq_without_usage_or_limit_headers_reports_what_it_has(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests, "post", lambda *a, **k: _ok('{"reply": "Hi"}'))
    client = GroqClient(api_key="k", model="m")
    client.chat(MESSAGES, SCHEMA)
    assert client.last_usage.prompt_tokens == 0 and client.last_usage.requests_per_day is None


def test_the_local_model_reports_its_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot_voice_assistant.ollama_client import OllamaClient

    body = {"message": {"content": '{"reply": "Hi"}'}, "prompt_eval_count": 900, "eval_count": 30}
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(200, body))
    client = OllamaClient.__new__(OllamaClient)
    client._chat_url, client._model, client._keep_alive, client._timeout_s = "http://x/api/chat", "m", "5m", 1.0
    client.chat(MESSAGES, SCHEMA)
    assert (client.last_usage.prompt_tokens, client.last_usage.completion_tokens) == (900, 30)


def test_the_fallback_reports_the_tokens_of_the_model_that_answered() -> None:
    from robot_voice_assistant.llm_backends import TokenUsage

    class Answering:
        def __init__(self, usage: TokenUsage, fail: bool = False) -> None:
            self.last_usage, self.fail = usage, fail

        def chat(self, *_args: object) -> str:
            if self.fail:
                raise requests.ConnectionError("down")
            return "{}"

    groq, llama = Answering(TokenUsage(1000, 20)), Answering(TokenUsage(800, 40))
    backend = FallbackBackend(groq, llama)
    backend.chat(MESSAGES, SCHEMA)
    assert backend.last_usage.prompt_tokens == 1000
    groq.fail = True
    backend.chat(MESSAGES, SCHEMA)
    assert backend.last_usage.prompt_tokens == 800


def test_without_a_local_model_a_short_rate_limit_is_waited_out() -> None:
    groq = FakeBackend("groq", [RateLimitedError(4.0), '{"reply": "Hi"}'])
    slept: list[float] = []
    backend = FallbackBackend(groq, None, sleep=slept.append)
    assert backend.chat(MESSAGES, SCHEMA) == '{"reply": "Hi"}'
    assert slept == [4.0] and backend.last_used == "groq" and groq.calls == 2


def test_without_a_local_model_a_long_rate_limit_or_an_error_fails() -> None:
    long_wait = FallbackBackend(FakeBackend("groq", [RateLimitedError(60.0)]), None, sleep=lambda _s: None)
    with pytest.raises(RateLimitedError):
        long_wait.chat(MESSAGES, SCHEMA)
    down = FallbackBackend(FakeBackend("groq", [requests.ConnectionError("down")]), None, sleep=lambda _s: None)
    with pytest.raises(requests.ConnectionError):
        down.chat(MESSAGES, SCHEMA)


def test_the_local_model_can_be_left_out() -> None:
    from robot_voice_assistant.assistant_factory import BrainSettings, create_backend

    backend, description = create_backend(BrainSettings(use_local_fallback=False), {"GROQ_API_KEY": "k"})
    assert not backend.has_fallback and "no local model" in description
    backend, _ = create_backend(BrainSettings(use_local_fallback=False), {})  # no Groq key: the local model is all
    assert backend.has_fallback
