"""Parsing of the LLM's structured reply into speech text and a gesture name."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

NO_GESTURE = "none"
NO_MOTION = "none"
FALLBACK_REPLY = "Sorry, I did not catch that."
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class AssistantResponse:
    reply: str
    gesture: str
    motion: str = NO_MOTION


def _load_json_object(raw: str) -> dict[str, object] | None:
    for candidate in (raw, *(_JSON_OBJECT.findall(raw))):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_assistant_response(
    raw: str, allowed_gestures: Sequence[str], allowed_motions: Sequence[str] = (NO_MOTION,)
) -> AssistantResponse:
    """Turn raw LLM output into a reply, a known gesture and a known motion; never raises."""
    data = _load_json_object(raw)
    if data is None:
        text = raw.strip()
        return AssistantResponse(text or FALLBACK_REPLY, NO_GESTURE)

    reply = str(data.get("reply", "")).strip() or FALLBACK_REPLY
    gesture = str(data.get("gesture", NO_GESTURE)).strip().lower()
    if gesture not in allowed_gestures:
        gesture = NO_GESTURE
    motion = str(data.get("motion", NO_MOTION)).strip().lower()
    if motion not in allowed_motions:
        motion = NO_MOTION
    return AssistantResponse(reply, gesture, motion)
