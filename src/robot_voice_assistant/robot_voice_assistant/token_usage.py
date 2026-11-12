"""How many tokens one answer used, and what is left of Groq's limits after it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0  # read: the system prompt, the conversation so far and the question
    completion_tokens: int = 0  # written: the JSON answer
    requests_per_day: int | None = None  # Groq's limits for this key and model, from its response headers
    requests_left_today: int | None = None
    tokens_per_minute: int | None = None
    tokens_left_this_minute: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> str:
        line = f"{self.prompt_tokens} in + {self.completion_tokens} out = {self.total_tokens} tokens"
        if self.requests_left_today is not None:
            line += f"; Groq: {self.requests_left_today} of {self.requests_per_day} requests left today"
        if self.tokens_left_this_minute is not None:
            line += f", {self.tokens_left_this_minute} of {self.tokens_per_minute} tokens left this minute"
        return line


def _number(headers: Mapping[str, str], name: str) -> int | None:
    try:
        return int(float(headers[name]))
    except (KeyError, TypeError, ValueError):
        return None


def groq_usage(body: Mapping[str, object], headers: Mapping[str, str]) -> TokenUsage:
    """From a Groq chat response: ``usage`` in the body, the rate limits in the ``x-ratelimit-*`` headers (the
    request limit is per day, the token limit per minute)."""
    usage = body.get("usage") if isinstance(body.get("usage"), Mapping) else {}
    return TokenUsage(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)),
                      _number(headers, "x-ratelimit-limit-requests"), _number(headers, "x-ratelimit-remaining-requests"),
                      _number(headers, "x-ratelimit-limit-tokens"), _number(headers, "x-ratelimit-remaining-tokens"))
