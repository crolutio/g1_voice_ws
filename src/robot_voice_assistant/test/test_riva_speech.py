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


ARABIC_VOICES = {"en": ("Magpie-Multilingual.EN-US.Ray", "en-US"), "ar": ("Magpie-Multilingual.AR-XA.Ray", "ar-XA")}


def test_each_piece_is_spoken_with_the_voice_of_its_language() -> None:
    service = FakeRivaService()
    engine = RivaSpeechEngine(service=service, voices=ARABIC_VOICES)
    engine.synthesize("Hello there.")
    engine.synthesize("مرحبا بك.")
    assert [(call["voice_name"], call["language_code"]) for call in service.calls] == [
        ("Magpie-Multilingual.EN-US.Ray", "en-US"), ("Magpie-Multilingual.AR-XA.Ray", "ar-XA")]


class FakeLocalVoice:
    sample_rate = 44100

    def __init__(self) -> None:
        self.texts: list[str] = []

    def synthesize(self, text: str) -> np.ndarray:
        self.texts.append(text)
        return np.full(4410, 1000, dtype=np.int16)


def test_when_the_cloud_fails_the_local_voice_apologises_once_instead_of_mangling_arabic() -> None:
    from robot_voice_assistant.speech_pipeline import NO_ARABIC_VOICE_APOLOGY

    service, local, now = FakeRivaService(fail=True), FakeLocalVoice(), [0.0]
    engine = FallbackSpeechEngine(RivaSpeechEngine(service=service, voices=ARABIC_VOICES), local,
                                  clock=lambda: now[0])
    assert len(engine.synthesize("مرحبا بك.")) > 0
    assert len(engine.synthesize("كيف يمكنني مساعدتك؟")) == 0  # the rest of the Arabic reply is skipped
    assert local.texts == [NO_ARABIC_VOICE_APOLOGY]
    assert len(engine.synthesize("Hello there.")) > 0  # English is still spoken by the local voice
    assert local.texts[-1] == "Hello there."


def test_arabic_tries_the_cloud_again_even_while_english_waits_for_it() -> None:
    service, local, now = FakeRivaService(fail=True), FakeLocalVoice(), [0.0]
    engine = FallbackSpeechEngine(RivaSpeechEngine(service=service, voices=ARABIC_VOICES), local,
                                  clock=lambda: now[0])
    engine.synthesize("Hello there.")  # the cloud fails: English uses the local voice for a while
    service.fail = False
    now[0] = 5.0
    engine.synthesize("مرحبا بك.")
    assert service.calls[-1]["language_code"] == "ar-XA"  # Arabic has no local voice, so it asked the cloud
    assert local.texts == ["Hello there."]
