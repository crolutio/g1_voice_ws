import threading
import time

import numpy as np

from robot_voice_assistant.speech_pipeline import SpeechPipeline, spoken_chunks


def test_reply_is_split_into_sentences_and_short_ones_are_joined() -> None:
    text = "Hi! I am G1. The capital of India is New Delhi, a large and diverse city with a rich history."
    assert spoken_chunks(text, min_chars=10) == [
        "Hi! I am G1.",
        "The capital of India is New Delhi, a large and diverse city with a rich history.",
    ]


def test_one_sentence_stays_one_chunk() -> None:
    assert spoken_chunks("Okay, stopping.") == ["Okay, stopping."]


class FakeEngine:
    sample_rate = 16000

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.texts: list[str] = []

    def synthesize(self, text: str) -> np.ndarray:
        time.sleep(self.delay_s)
        self.texts.append(text)
        return np.full(10, len(self.texts), dtype=np.int16)


class FakePlayer:
    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.played: list[int] = []
        self.started_at: list[float] = []

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        self.started_at.append(time.monotonic())
        time.sleep(self.delay_s)
        self.played.append(int(audio[0]))


def test_all_sentences_are_played_in_order() -> None:
    engine, player = FakeEngine(), FakePlayer()
    SpeechPipeline(engine, player).speak("First sentence here. Second sentence here. Third sentence here.", min_chars=5)
    assert engine.texts == ["First sentence here.", "Second sentence here.", "Third sentence here."]
    assert player.played == [1, 2, 3]


def test_playback_starts_before_the_whole_reply_is_synthesized() -> None:
    engine, player = FakeEngine(delay_s=0.2), FakePlayer(delay_s=0.2)
    started = time.monotonic()
    SpeechPipeline(engine, player).speak("One sentence. Two sentence. Three sentence.", min_chars=5)
    assert player.started_at[0] - started < 0.35  # first sentence plays after one synthesis, not three
    assert time.monotonic() - started < 0.9  # synthesis of later sentences overlaps playback


def test_a_failing_sentence_does_not_stop_the_rest() -> None:
    class FlakyEngine(FakeEngine):
        def synthesize(self, text: str) -> np.ndarray:
            if "bad" in text:
                raise RuntimeError("boom")
            return super().synthesize(text)

    player = FakePlayer()
    errors: list[str] = []
    SpeechPipeline(FlakyEngine(), player, on_error=errors.append).speak("A bad one here. A good one here.", min_chars=5)
    assert player.played == [1]
    assert errors and "boom" in errors[0]


def test_pipeline_can_be_used_from_several_threads_one_reply_at_a_time() -> None:
    engine, player = FakeEngine(), FakePlayer(delay_s=0.01)
    pipeline = SpeechPipeline(engine, player)
    threads = [threading.Thread(target=pipeline.speak, args=(f"Reply number {i} is here.",)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(player.played) == 3
