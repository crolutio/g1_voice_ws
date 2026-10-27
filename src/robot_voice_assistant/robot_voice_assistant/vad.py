"""Energy-based voice activity detection that splits a microphone stream into utterances."""

from __future__ import annotations

from collections import deque

import numpy as np


class EnergyVadSegmenter:
    """Collects fixed-size audio frames and returns complete utterances.

    The speech threshold follows the background noise floor: it is calibrated on the first
    frames, then tracked continuously (quickly downward, slowly upward), so steady noise such
    as a laptop fan is learned as background even when it is louder than ``min_rms``.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_ms: int = 30,
        threshold_ratio: float = 3.0,
        min_rms: float = 0.01,
        end_silence_ms: int = 700,
        min_speech_ms: int = 250,
        max_utterance_s: float = 15.0,
        pre_roll_ms: int = 300,
        calibration_ms: int = 500,
        floor_fall_rate: float = 0.1,
        floor_rise_rate_idle: float = 0.02,
        floor_rise_rate_speech: float = 0.001,
    ) -> None:
        self._threshold_ratio = threshold_ratio
        self._min_rms = min_rms
        self._end_silence_frames = max(1, round(end_silence_ms / frame_ms))
        self._frame_ms = frame_ms
        self._min_speech_frames = max(1, round(min_speech_ms / frame_ms))
        self._max_utterance_samples = int(max_utterance_s * sample_rate)
        self._calibration_frames = max(1, round(calibration_ms / frame_ms))
        self._floor_fall_rate = floor_fall_rate
        self._floor_rise_rate_idle = floor_rise_rate_idle
        self._floor_rise_rate_speech = floor_rise_rate_speech
        self._pre_roll: deque[np.ndarray] = deque(maxlen=max(1, round(pre_roll_ms / frame_ms)))
        self._calibration_rms: list[float] = []
        self._noise_floor = 0.0
        self._active_frames: list[np.ndarray] = []
        self._speech_frame_count = 0
        self._trailing_silence_frames = 0
        self._in_utterance = False
        self.last_rms = 0.0

    @property
    def is_calibrated(self) -> bool:
        return len(self._calibration_rms) >= self._calibration_frames

    @property
    def end_silence_s(self) -> float:
        """How long after the speech stops an utterance is returned."""
        return self._end_silence_frames * self._frame_ms / 1000

    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    @property
    def threshold(self) -> float:
        return max(self._min_rms, self._noise_floor * self._threshold_ratio)

    def reset(self) -> None:
        """Drop any partially collected utterance (e.g. while the robot itself is speaking)."""
        self._active_frames = []
        self._pre_roll.clear()
        self._speech_frame_count = 0
        self._trailing_silence_frames = 0
        self._in_utterance = False

    def _track_noise_floor(self, rms: float) -> None:
        if rms < self._noise_floor:
            rate = self._floor_fall_rate
        else:
            rate = self._floor_rise_rate_speech if self._in_utterance else self._floor_rise_rate_idle
        self._noise_floor += rate * (rms - self._noise_floor)

    def push(self, frame: np.ndarray) -> np.ndarray | None:
        """Add one float32 frame in [-1, 1]; return an utterance when one has just ended."""
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if frame.size else 0.0
        self.last_rms = rms

        if not self.is_calibrated:
            self._calibration_rms.append(rms)
            self._noise_floor = float(np.median(self._calibration_rms))
            self._pre_roll.append(frame)
            return None

        is_speech = rms >= self.threshold
        self._track_noise_floor(rms)

        if not self._in_utterance:
            if not is_speech:
                self._pre_roll.append(frame)
                return None
            self._in_utterance = True
            self._active_frames = list(self._pre_roll)
            self._pre_roll.clear()
            self._speech_frame_count = 0
            self._trailing_silence_frames = 0

        self._active_frames.append(frame)
        if is_speech:
            self._speech_frame_count += 1
            self._trailing_silence_frames = 0
        else:
            self._trailing_silence_frames += 1

        collected_samples = sum(len(f) for f in self._active_frames)
        ended_by_silence = self._trailing_silence_frames >= self._end_silence_frames
        ended_by_length = collected_samples >= self._max_utterance_samples
        if not (ended_by_silence or ended_by_length):
            return None

        utterance = np.concatenate(self._active_frames).astype(np.float32)
        long_enough = self._speech_frame_count >= self._min_speech_frames
        self.reset()
        return utterance if long_enough else None
