from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robot_voice_assistant.speech_recognition import Transcriber, vocabulary_prompt


class FakeWhisper:
    def __init__(self, texts: list[str], no_speech: float = 0.1) -> None:
        self.texts = texts
        self.no_speech = no_speech
        self.options: dict[str, Any] = {}

    def transcribe(self, audio: np.ndarray, **options: Any) -> tuple[list[SimpleNamespace], None]:
        self.options = options
        return [SimpleNamespace(text=text, no_speech_prob=self.no_speech) for text in self.texts], None


def test_segments_are_joined_into_one_transcript() -> None:
    whisper = FakeWhisper([" Robot, what is", " the capital of India?"])
    transcript = Transcriber(whisper).transcribe(np.zeros(1600, dtype=np.float32))
    assert transcript.text == "Robot, what is the capital of India?"
    assert transcript.no_speech_probability == 0.1
    assert transcript.seconds >= 0.0


def test_no_segments_means_no_speech() -> None:
    transcript = Transcriber(FakeWhisper([])).transcribe(np.zeros(1600, dtype=np.float32))
    assert transcript.text == ""
    assert transcript.no_speech_probability == 1.0


def test_vocabulary_is_given_as_a_prompt_by_default() -> None:
    whisper = FakeWhisper(["hi"])
    Transcriber(whisper, vocabulary=("Aideology",)).transcribe(np.zeros(16, dtype=np.float32))
    assert "Aideology" in whisper.options["initial_prompt"]
    assert "hotwords" not in whisper.options


def test_vocabulary_can_be_given_as_hotwords_or_not_at_all() -> None:
    whisper = FakeWhisper(["hi"])
    Transcriber(whisper, vocabulary=("Aideology", "R1"), vocabulary_as="hotwords").transcribe(np.zeros(16))
    assert whisper.options["hotwords"] == "Aideology R1"
    Transcriber(whisper, vocabulary=("Aideology",), vocabulary_as="none").transcribe(np.zeros(16))
    assert "hotwords" not in whisper.options and "initial_prompt" not in whisper.options


def test_unknown_vocabulary_option_is_rejected() -> None:
    with pytest.raises(ValueError):
        Transcriber(FakeWhisper([]), vocabulary_as="magic")


def test_vocabulary_prompt_names_every_word() -> None:
    prompt = vocabulary_prompt(("Aideology", "Jorge Colotto"))
    assert "Aideology" in prompt and "Jorge Colotto" in prompt
    # No commands: Whisper sometimes copies prompt words into noisy speech, and a copied command would move the robot
    assert "walk" not in prompt.lower() and "turn" not in prompt.lower() and "robot" not in prompt.lower()
    assert vocabulary_prompt(()) == ""
