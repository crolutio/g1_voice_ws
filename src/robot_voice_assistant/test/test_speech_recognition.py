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


class PromptEchoWhisper:
    """Whisper that copies its prompt instead of the speech when given one (the failure seen live)."""

    def __init__(self, speech: str) -> None:
        self.speech = speech
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: np.ndarray, **options: Any) -> tuple[list[SimpleNamespace], None]:
        self.calls.append(options)
        text = options.get("initial_prompt") or self.speech
        return [SimpleNamespace(text=" " + text, no_speech_prob=0.05)], None


def test_a_transcript_that_only_repeats_the_prompt_is_transcribed_again_without_it() -> None:
    whisper = PromptEchoWhisper("What is the weather like today?")
    vocabulary = ("Aideology", "Jorge Colotto", "Dubai Silicon Oasis", "MDS")
    transcript = Transcriber(whisper, vocabulary=vocabulary).transcribe(np.zeros(16000, dtype=np.float32))
    assert transcript.text == "What is the weather like today?"
    assert len(whisper.calls) == 2 and "initial_prompt" not in whisper.calls[1]


def test_a_real_question_using_the_names_is_kept() -> None:
    whisper = FakeWhisper([" Who is Jorge Colotto at Aideology?"])
    transcript = Transcriber(whisper, vocabulary=("Aideology", "Jorge Colotto", "MDS")).transcribe(np.zeros(16))
    assert transcript.text == "Who is Jorge Colotto at Aideology?"


def test_a_partial_echo_of_the_prompt_also_counts() -> None:
    whisper = FakeWhisper([" Aideology, Jorge Colotto."])
    vocabulary = ("Aideology", "Jorge Colotto", "Dubai Silicon Oasis", "MDS")
    from robot_voice_assistant.speech_recognition import is_prompt_echo

    assert is_prompt_echo("Aideology, Jorge Colotto.", vocabulary)
    assert is_prompt_echo("aideology jorge colotto dubai silicon oasis mds", vocabulary)
    assert not is_prompt_echo("Tell me about Aideology.", vocabulary)
    assert not is_prompt_echo("", vocabulary)
