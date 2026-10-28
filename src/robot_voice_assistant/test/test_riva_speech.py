import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robot_voice_assistant.speech_pipeline import FallbackSpeechEngine, RivaSpeechEngine


class FakeRivaService:
    def __init__(self, fail: bool = False, delay_s: float = 0.0) -> None:
        self.fail = fail
        self.delay_s = delay_s
        self.calls: list[dict[str, Any]] = []

    def synthesize(self, text: str, **options: Any) -> SimpleNamespace:
        self.calls.append({"text": text, **options})
        if self.delay_s:
            threading.Event().wait(self.delay_s)
        if self.fail:
            raise ConnectionError("NVIDIA cloud unreachable")
        return SimpleNamespace(audio=np.arange(8, dtype=np.int16).tobytes())


def test_riva_returns_16_bit_audio_for_the_chosen_voice() -> None:
    service = FakeRivaService()
    engine = RivaSpeechEngine(service=service, voice="Magpie-Multilingual.EN-US.Aria", sample_rate=44100)
    audio = engine.synthesize("Hello there.")
    assert audio.dtype == np.int16 and list(audio) == list(range(8))
    assert engine.sample_rate == 44100
    call = service.calls[0]
    assert call["text"] == "Hello there."
    assert call["voice_name"] == "Magpie-Multilingual.EN-US.Aria"
    assert call["language_code"] == "en-US"
    assert call["sample_rate_hz"] == 44100


def test_a_slow_cloud_answer_counts_as_a_failure() -> None:
    engine = RivaSpeechEngine(service=FakeRivaService(delay_s=0.5), timeout_s=0.1)
    with pytest.raises(TimeoutError):
        engine.synthesize("Hello.")


class FakeEngine:
    def __init__(self, sample_rate: int, fail: bool = False) -> None:
        self.sample_rate = sample_rate
        self.fail = fail
        self.texts: list[str] = []

    def synthesize(self, text: str) -> np.ndarray:
        self.texts.append(text)
        if self.fail:
            raise ConnectionError("down")
        return np.full(self.sample_rate, 1000, dtype=np.int16)  # one second


def test_the_cloud_voice_is_used_while_it_works() -> None:
    primary, fallback = FakeEngine(44100), FakeEngine(24000)
    engine = FallbackSpeechEngine(primary, fallback)
    assert len(engine.synthesize("Hi.")) == 44100
    assert engine.sample_rate == 44100 and fallback.texts == []


def test_the_local_voice_speaks_when_the_cloud_fails_at_the_same_sample_rate() -> None:
    notes: list[str] = []
    engine = FallbackSpeechEngine(FakeEngine(44100, fail=True), FakeEngine(24000), on_fallback=notes.append)
    audio = engine.synthesize("Hi.")
    assert len(audio) == pytest.approx(44100, abs=2)  # resampled from 24 kHz, still one second
    assert notes and "down" in notes[0]


def test_the_local_voice_can_be_sped_up_to_match_the_cloud_voice() -> None:
    engine = FallbackSpeechEngine(FakeEngine(44100, fail=True), FakeEngine(24000), fallback_speed=1.25)
    assert len(engine.synthesize("Hi.")) == pytest.approx(44100 / 1.25, rel=0.05)


def test_after_a_failure_the_cloud_is_skipped_for_a_while() -> None:
    clock = [100.0]
    primary, fallback = FakeEngine(44100, fail=True), FakeEngine(24000)
    engine = FallbackSpeechEngine(primary, fallback, cooldown_s=30.0, clock=lambda: clock[0])
    engine.synthesize("One.")
    engine.synthesize("Two.")
    assert primary.texts == ["One."]  # not tried again during the cooldown
    clock[0] += 31.0
    primary.fail = False
    engine.synthesize("Three.")
    assert primary.texts == ["One.", "Three."]
