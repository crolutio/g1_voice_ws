"""Speech to text with Whisper, told which names to expect so it spells them right ("Aideology", not "ideology").

With several languages (English and Arabic), Whisper first detects which one was spoken, then writes it down in it.
A sentence that mixes them is written down in each, and the version Whisper is surer of is kept: told "English", it
drops or garbles the Arabic words ("لوح لي please" became "Lowly, please."); told "Arabic", it usually keeps both.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_voice_assistant.language import ENGLISH, choose_language

# A sentence counts as mixed when the second language gets at least this share of the detection: in the tests,
# sentences in one language gave it at most 0.007, sentences mixing English and Arabic at least 0.014.
MIXED_LANGUAGE_SHARE = 0.01

# Names Whisper does not know and would otherwise mishear.
DEFAULT_VOCABULARY = ("Aideology", "Jorge Colotto", "Dubai Silicon Oasis", "MDS")


@dataclass(frozen=True)
class Transcript:
    text: str
    no_speech_probability: float
    seconds: float  # how long transcribing took
    language: str = ENGLISH  # what was spoken ("en", "ar")


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
        languages: Sequence[str] = (),
    ) -> None:
        """``model`` is a faster_whisper.WhisperModel. ``vocabulary_as``: "prompt", "hotwords" or "none".

        ``languages``: what people may speak, e.g. ("en", "ar"); with more than one, each sentence's language is
        detected first (empty = ``language`` only). The names are given to Whisper for English only: prompted with
        English names, Whisper tends to answer Arabic speech in English instead of writing it down.
        """
        self._model = model
        self._vocabulary = tuple(vocabulary)
        self._languages = tuple(languages) or (language,)
        self._options: dict[str, Any] = {"beam_size": beam_size}
        if vocabulary and vocabulary_as == "prompt":
            self._options["initial_prompt"] = vocabulary_prompt(vocabulary)
        elif vocabulary and vocabulary_as == "hotwords":
            self._options["hotwords"] = " ".join(vocabulary)
        elif vocabulary_as not in ("prompt", "hotwords", "none"):
            raise ValueError(f"vocabulary_as must be prompt, hotwords or none, not {vocabulary_as!r}")

    def transcribe(self, audio: np.ndarray) -> Transcript:
        started = time.perf_counter()
        versions = [self._transcribe_in(audio, language) for language in self._spoken_languages(audio)]
        text, no_speech, _, language = max(versions, key=lambda version: version[2])  # the surest
        return Transcript(text, no_speech, time.perf_counter() - started, language)

    def _spoken_languages(self, audio: np.ndarray) -> list[str]:
        """The detected language, or two when the sentence seems to mix them."""
        if len(self._languages) == 1:
            return [self._languages[0]]
        _, _, probabilities = self._model.detect_language(audio)
        first = choose_language(probabilities, self._languages)
        scores = dict(probabilities)
        others = [language for language in self._languages if language != first]
        second = max(others, key=lambda language: scores.get(language, 0.0))
        total = scores.get(first, 0.0) + scores.get(second, 0.0)
        mixed = total > 0 and scores.get(second, 0.0) / total >= MIXED_LANGUAGE_SHARE
        return [first, second] if mixed else [first]

    def _transcribe_in(self, audio: np.ndarray, language: str) -> tuple[str, float, float, str]:
        """(text, no-speech probability, confidence, language)."""
        options = {**self._options, "language": language}
        if language != ENGLISH:
            options = {key: value for key, value in options.items() if key not in ("initial_prompt", "hotwords")}
        text, no_speech, confidence = self._run(audio, options)
        if "initial_prompt" in options and is_prompt_echo(text, self._vocabulary):
            # Whisper repeated the name list instead of the speech: listen again without the list
            without_prompt = {key: value for key, value in options.items() if key != "initial_prompt"}
            text, no_speech, confidence = self._run(audio, without_prompt)
        return text, no_speech, confidence, language

    def _run(self, audio: np.ndarray, options: dict[str, Any]) -> tuple[str, float, float]:
        """(text, no-speech probability, confidence: the mean log probability of the words)."""
        segments = list(self._model.transcribe(audio, **options)[0])
        text = "".join(segment.text for segment in segments).strip()
        confidence = sum(segment.avg_logprob for segment in segments) / len(segments) if segments else float("-inf")
        return text, max((segment.no_speech_prob for segment in segments), default=1.0), confidence
