"""Decides which transcripts are addressed to the robot, so nearby conversations are ignored."""

from __future__ import annotations

import re
from collections.abc import Sequence

GREETING_FOR_BARE_WAKE_WORD = "Hello!"


def _wake_word_pattern(wake_word: str) -> re.Pattern[str]:
    words = [re.escape(word) for word in wake_word.lower().split()]
    # Whole words only, allowing spaces or hyphens between parts ("g one", "g-one").
    return re.compile(r"\b" + r"[\s\-]*".join(words) + r"\b", re.IGNORECASE)


class WakeWordGate:
    def __init__(self, wake_words: Sequence[str], follow_up_s: float) -> None:
        self._patterns = [_wake_word_pattern(w) for w in wake_words if w.strip()]
        self._follow_up_s = follow_up_s
        self._follow_up_until_s = float("-inf")

    def note_reply_finished(self, now_s: float) -> None:
        """Open a short window in which the user can follow up without the wake word."""
        self._follow_up_until_s = now_s + self._follow_up_s

    def accept(self, transcript: str, now_s: float) -> str | None:
        """Return the request text if it is addressed to the robot, otherwise None."""
        text = transcript.strip()
        if not self._patterns or now_s <= self._follow_up_until_s:
            return text or None
        for pattern in self._patterns:
            match = pattern.search(text)
            if match is None:
                continue
            remainder = (text[: match.start()] + " " + text[match.end() :]).strip()
            remainder = re.sub(r"^(hey|hi|hello|ok|okay)\b", "", remainder, flags=re.IGNORECASE)
            remainder = remainder.lstrip(" ,.!?-").rstrip()
            if not re.search(r"\w", remainder):
                return GREETING_FOR_BARE_WAKE_WORD
            return remainder
        return None
