"""Rejects Whisper output that is almost certainly noise rather than speech."""

from __future__ import annotations

import re

# Phrases Whisper tends to invent on silence, breathing or keyboard noise.
_HALLUCINATIONS = frozenset(
    {
        "",
        "you",
        "thank you",
        "thanks for watching",
        "thank you for watching",
        "bye",
        "okay",
        "so",
        "uh",
        "um",
        "hmm",
    }
)
_NON_WORD = re.compile(r"[^a-z0-9' ]+")


def normalize(text: str) -> str:
    return " ".join(_NON_WORD.sub(" ", text.lower()).split())


def is_babble(normalized: str) -> bool:
    """Whisper's output for noise or mumbling: one word repeated ("ah, ah, ah, ah")."""
    words = normalized.replace("-", " ").split()
    if len(words) < 4:
        return False
    one_word = max(words.count(word) for word in set(words)) / len(words) >= 0.75  # "ah, ah, ah, ah"
    few_words = len(words) >= 8 and len(set(words)) / len(words) <= 0.25  # "uh-huh, uh-huh, uh-huh, ..."
    return one_word or few_words


def is_meaningful_transcript(text: str, no_speech_probability: float, max_no_speech: float = 0.6) -> bool:
    if no_speech_probability > max_no_speech:
        return False
    normalized = normalize(text)
    if normalized in _HALLUCINATIONS or is_babble(normalized):
        return False
    return any(character.isalpha() for character in normalized)
