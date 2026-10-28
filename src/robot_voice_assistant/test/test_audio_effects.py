import numpy as np
import pytest

from robot_voice_assistant.audio_effects import change_speed, trim_silence

RATE = 24000


def tone(frequency: float, seconds: float, level: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (np.sin(2 * np.pi * frequency * t) * level * 32767).astype(np.int16)


def main_frequency(audio: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(audio.astype(np.float64) * np.hanning(len(audio))))
    return float(np.fft.rfftfreq(len(audio), 1 / RATE)[np.argmax(spectrum)])


def test_faster_speech_is_shorter_by_the_rate() -> None:
    audio = tone(200, 3.0)
    faster = change_speed(audio, RATE, 1.15)
    assert len(faster) / RATE == pytest.approx(3.0 / 1.15, rel=0.03)
    assert faster.dtype == np.int16


def test_faster_speech_keeps_its_pitch() -> None:
    faster = change_speed(tone(200, 3.0), RATE, 1.15)
    assert main_frequency(faster[RATE // 4 : -RATE // 4]) == pytest.approx(200, abs=3)


def test_faster_speech_keeps_its_loudness_without_clicks() -> None:
    faster = change_speed(tone(200, 3.0, level=0.5), RATE, 1.15).astype(np.float64) / 32767
    middle = faster[RATE // 4 : -RATE // 4]
    assert np.sqrt(np.mean(middle**2)) == pytest.approx(0.5 / np.sqrt(2), rel=0.1)
    assert np.max(np.abs(np.diff(middle))) < 0.1  # no jumps between the joined pieces


def test_normal_speed_returns_the_audio_unchanged() -> None:
    audio = tone(200, 1.0)
    assert np.array_equal(change_speed(audio, RATE, 1.0), audio)


def test_very_short_audio_is_returned_unchanged() -> None:
    audio = tone(200, 0.01)
    assert np.array_equal(change_speed(audio, RATE, 1.2), audio)


def test_silence_around_speech_is_trimmed_to_a_short_pause() -> None:
    silence = np.zeros(int(0.5 * RATE), dtype=np.int16)
    audio = np.concatenate([silence, tone(200, 1.0), silence])
    trimmed = trim_silence(audio, RATE, keep_before_s=0.05, keep_after_s=0.12)
    assert len(trimmed) / RATE == pytest.approx(1.0 + 0.05 + 0.12, abs=0.02)


def test_quiet_room_noise_counts_as_silence() -> None:
    noise = (np.random.default_rng(0).standard_normal(int(0.5 * RATE)) * 100).astype(np.int16)
    audio = np.concatenate([noise, tone(200, 1.0), noise])
    assert len(trim_silence(audio, RATE)) / RATE < 1.3


def test_all_silent_audio_becomes_empty() -> None:
    assert len(trim_silence(np.zeros(RATE, dtype=np.int16), RATE)) == 0
