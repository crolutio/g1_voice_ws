"""Speech to text with Whisper, told which names to expect so it spells them right ("Aideology", not "ideology")."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# Names Whisper does not know and would otherwise mishear.
DEFAULT_VOCABULARY = ("Aideology", "Jorge Colotto", "Dubai Silicon Oasis", "MDS")


@dataclass(frozen=True)
class Transcript:
    text: str
    no_speech_probability: float
    seconds: float  # how long transcribing took


def vocabulary_prompt(vocabulary: Sequence[str]) -> str:
    """Words Whisper then spells the same way. Only names: Whisper sometimes copies prompt words into short or
    noisy speech, which is harmless for a name but would move the robot for a command such as "turn right"."""
    return f"{', '.join(vocabulary)}." if vocabulary else ""


def is_prompt_echo(text: str, vocabulary: Sequence[str]) -> bool:
    """True when ``text`` is only words from the prompt (Whisper copied the prompt instead of the speech)."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    prompt_words = set(re.findall(r"[a-z0-9']+", " ".join(vocabulary).lower()))
    return bool(words) and all(word in prompt_words for word in words)


class Transcriber:
    def __init__(
        self,
        model: Any,
        language: str = "en",
        beam_size: int = 5,
        vocabulary: Sequence[str] = DEFAULT_VOCABULARY,
        vocabulary_as: str = "prompt",
    ) -> None:
        """``model`` is a faster_whisper.WhisperModel. ``vocabulary_as``: "prompt", "hotwords" or "none"."""
        self._model = model
        self._vocabulary = tuple(vocabulary)
        self._options: dict[str, Any] = {"language": language, "beam_size": beam_size}
        if vocabulary and vocabulary_as == "prompt":
            self._options["initial_prompt"] = vocabulary_prompt(vocabulary)
        elif vocabulary and vocabulary_as == "hotwords":
            self._options["hotwords"] = " ".join(vocabulary)
        elif vocabulary_as not in ("prompt", "hotwords", "none"):
            raise ValueError(f"vocabulary_as must be prompt, hotwords or none, not {vocabulary_as!r}")

    def transcribe(self, audio: np.ndarray) -> Transcript:
        started = time.perf_counter()
        text, no_speech = self._run(audio, self._options)
        if "initial_prompt" in self._options and is_prompt_echo(text, self._vocabulary):
            # Whisper repeated the name list instead of the speech: listen again without the list
            without_prompt = {key: value for key, value in self._options.items() if key != "initial_prompt"}
            text, no_speech = self._run(audio, without_prompt)
        return Transcript(text, no_speech, time.perf_counter() - started)

    def _run(self, audio: np.ndarray, options: dict[str, Any]) -> tuple[str, float]:
        segments = list(self._model.transcribe(audio, **options)[0])
        text = "".join(segment.text for segment in segments).strip()
        return text, max((segment.no_speech_prob for segment in segments), default=1.0)
