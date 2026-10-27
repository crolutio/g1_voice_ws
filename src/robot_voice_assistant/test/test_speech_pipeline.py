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


def test_first_sentence_is_spoken_on_its_own_so_the_voice_starts_early() -> None:
    text = "The capital of India is New Delhi. It is a planned city built to replace Calcutta as the capital in the 1930s."
    assert spoken_chunks(text) == [
        "The capital of India is New Delhi.",
        "It is a planned city built to replace Calcutta as the capital in the 1930s.",
    ]


def test_a_long_first_sentence_is_split_at_a_clause() -> None:
    text = (
        "Aideology is an AI infrastructure company based in Dubai that helps organizations move AI projects "
        "from pilot to production using GPU-accelerated computing."
    )
    assert spoken_chunks(text) == [
        "Aideology is an AI infrastructure company based in Dubai",
        "that helps organizations move AI projects from pilot to production using GPU-accelerated computing.",
    ]


def test_long_sentences_are_split_at_clauses_and_commas() -> None:
    text = (
        "Sure. The sky looks blue because sunlight scatters off the air molecules in the atmosphere, "
        "and blue light scatters much more strongly than red light does."
    )
    assert spoken_chunks(text) == [
        "Sure. The sky looks blue",
        "because sunlight scatters off the air molecules in the atmosphere,",
        "and blue light scatters much more strongly than red light does.",
    ]


def test_a_long_first_sentence_is_cut_at_the_nearest_break_past_the_limit() -> None:
    text = (
        "The sky looks blue because sunlight scatters in the atmosphere, and blue light scatters more than other "
        "colors. This effect is called Rayleigh scattering."
    )
    assert spoken_chunks(text) == [
        "The sky looks blue because sunlight scatters in the atmosphere,",
        "and blue light scatters more than other colors.",
        "This effect is called Rayleigh scattering.",
    ]


def test_a_sentence_without_a_good_break_is_not_cut() -> None:
    text = "Supercalifragilisticexpialidocious-is-a-very-long-made-up-word-that-has-no-spaces-in-it-at-all-anywhere."
    assert spoken_chunks(text) == [text]


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


def test_speaking_reports_when_the_voice_started_and_how_long_each_piece_took() -> None:
    engine, player = FakeEngine(delay_s=0.05), FakePlayer(delay_s=0.05)
    first_audio: list[float] = []
    timing = SpeechPipeline(engine, player).speak(
        "First sentence here. Second sentence here.", min_chars=5, on_first_audio=first_audio.append
    )
    assert len(first_audio) == 1
    assert first_audio[0] == timing.first_audio_s
    assert 0.04 < timing.first_audio_s < 0.2
    assert [chars for chars, _ in timing.pieces] == [len("First sentence here."), len("Second sentence here.")]
    assert all(0.04 < seconds < 0.2 for _, seconds in timing.pieces)
    assert timing.total_s >= timing.first_audio_s + 0.1  # both pieces were played


def test_nothing_spoken_reports_no_first_audio() -> None:
    timing = SpeechPipeline(FakeEngine(), FakePlayer()).speak("   ")
    assert timing.first_audio_s is None
    assert timing.pieces == []
