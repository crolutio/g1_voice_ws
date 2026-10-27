"""Speech to text with Whisper, told which names to expect so it spells them right ("Aideology", not "ideology")."""

from __future__ import annotations

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
        self._options: dict[str, Any] = {"language": language, "beam_size": beam_size}
        if vocabulary and vocabulary_as == "prompt":
            self._options["initial_prompt"] = vocabulary_prompt(vocabulary)
        elif vocabulary and vocabulary_as == "hotwords":
            self._options["hotwords"] = " ".join(vocabulary)
        elif vocabulary_as not in ("prompt", "hotwords", "none"):
            raise ValueError(f"vocabulary_as must be prompt, hotwords or none, not {vocabulary_as!r}")

    def transcribe(self, audio: np.ndarray) -> Transcript:
        started = time.perf_counter()
        segments = list(self._model.transcribe(audio, **self._options)[0])
        text = "".join(segment.text for segment in segments).strip()
        no_speech = max((segment.no_speech_prob for segment in segments), default=1.0)
        return Transcript(text, no_speech, time.perf_counter() - started)
