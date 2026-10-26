"""Minimal client for a local Ollama server's chat endpoint."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import requests


class ChatMessage(TypedDict):
    role: str
    content: str


def build_response_schema(
    gesture_names: Sequence[str], motion_names: Sequence[str] | None = None
) -> dict[str, object]:
    """JSON schema that constrains Ollama to a reply, one known gesture and (when walking) one motion."""
    properties: dict[str, object] = {
        "reply": {"type": "string"},
        "gesture": {"type": "string", "enum": list(gesture_names)},
    }
    if motion_names:
        properties["motion"] = {"type": "string", "enum": list(motion_names)}
    return {"type": "object", "properties": properties, "required": list(properties)}


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 60.0, keep_alive: str = "30m") -> None:
        self._chat_url = f"{base_url.rstrip('/')}/api/chat"
        self._model = model
        self._timeout_s = timeout_s
        self._keep_alive = keep_alive

    def warm_up(self) -> None:
        """Load the model into GPU memory now so the first real question answers quickly."""
        requests.post(
            self._chat_url,
            json={"model": self._model, "messages": [], "keep_alive": self._keep_alive},
            timeout=self._timeout_s * 3,
        ).raise_for_status()

    def chat(self, messages: Sequence[ChatMessage], response_schema: dict[str, object]) -> str:
        """Send the conversation and return the assistant's raw message content."""
        response = requests.post(
            self._chat_url,
            json={
                "model": self._model,
                "messages": list(messages),
                "format": response_schema,
                "stream": False,
                "keep_alive": self._keep_alive,
                "options": {"temperature": 0.4, "num_predict": 300},
            },
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        return str(response.json()["message"]["content"])
