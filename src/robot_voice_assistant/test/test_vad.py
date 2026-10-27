import pytest
import numpy as np

from robot_voice_assistant.vad import EnergyVadSegmenter

SAMPLE_RATE = 16000
FRAME_SAMPLES = 480  # 30 ms


def _frames(amplitude: float, duration_s: float, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    total = int(duration_s * SAMPLE_RATE)
    signal = (rng.standard_normal(total) * amplitude).astype(np.float32)
    return [signal[i : i + FRAME_SAMPLES] for i in range(0, total - FRAME_SAMPLES + 1, FRAME_SAMPLES)]


def _feed(segmenter: EnergyVadSegmenter, frames: list[np.ndarray]) -> list[np.ndarray]:
    utterances = []
    for frame in frames:
        utterance = segmenter.push(frame)
        if utterance is not None:
            utterances.append(utterance)
    return utterances


def test_silence_produces_no_utterance() -> None:
    segmenter = EnergyVadSegmenter(sample_rate=SAMPLE_RATE)
    assert _feed(segmenter, _frames(0.001, 3.0)) == []


def test_speech_followed_by_silence_produces_one_utterance() -> None:
    segmenter = EnergyVadSegmenter(sample_rate=SAMPLE_RATE)
    frames = _frames(0.001, 1.0, seed=1) + _frames(0.2, 1.0, seed=2) + _frames(0.001, 1.5, seed=3)
    utterances = _feed(segmenter, frames)
    assert len(utterances) == 1
    duration_s = len(utterances[0]) / SAMPLE_RATE
    # One second of speech plus pre-roll and the trailing silence window.
    assert 1.0 <= duration_s <= 2.2


def test_short_click_is_ignored() -> None:
    segmenter = EnergyVadSegmenter(sample_rate=SAMPLE_RATE)
    frames = _frames(0.001, 1.0, seed=1) + _frames(0.3, 0.06, seed=2) + _frames(0.001, 1.5, seed=3)
    assert _feed(segmenter, frames) == []


def test_long_speech_is_cut_at_max_duration() -> None:
    segmenter = EnergyVadSegmenter(sample_rate=SAMPLE_RATE, max_utterance_s=2.0)
    frames = _frames(0.001, 0.5, seed=1) + _frames(0.2, 5.0, seed=2)
    utterances = _feed(segmenter, frames)
    assert len(utterances) >= 2
    assert all(len(u) / SAMPLE_RATE <= 2.5 for u in utterances)


def test_loud_steady_noise_is_learned_as_background() -> None:
    # Fan noise louder than min_rms right from the start must not count as speech.
    segmenter = EnergyVadSegmenter(sample_rate=SAMPLE_RATE, min_rms=0.01)
    assert _feed(segmenter, _frames(0.05, 20.0, seed=1)) == []
    assert segmenter.threshold > 0.05


def test_speech_over_loud_noise_is_detected() -> None:
    segmenter = EnergyVadSegmenter(sample_rate=SAMPLE_RATE, min_rms=0.01)
    rng = np.random.default_rng(7)
    noise = _frames(0.05, 4.5, seed=1)
    speech = [f + (rng.standard_normal(len(f)) * 0.3).astype(np.float32) for f in _frames(0.05, 1.2, seed=2)]
    utterances = _feed(segmenter, noise[:100] + speech + noise[100:])
    assert len(utterances) == 1


def test_reset_discards_partial_utterance() -> None:
    segmenter = EnergyVadSegmenter(sample_rate=SAMPLE_RATE)
    _feed(segmenter, _frames(0.001, 1.0, seed=1) + _frames(0.2, 0.6, seed=2))
    segmenter.reset()
    assert _feed(segmenter, _frames(0.001, 1.5, seed=3)) == []


def test_reports_how_long_it_waits_for_silence_before_ending_an_utterance() -> None:
    assert EnergyVadSegmenter(sample_rate=16000, frame_ms=30, end_silence_ms=700).end_silence_s == pytest.approx(0.69)  # 23 frames of 30 ms
