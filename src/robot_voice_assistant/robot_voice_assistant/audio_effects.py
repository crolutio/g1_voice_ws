"""Audio clean-up for the robot's voice: trim the silence around each spoken piece and speed speech up.

Qwen3-TTS (0.6B) speaks at about 140 words per minute, ignores style instructions such as "speak faster",
and adds about half a second of silence before and after every piece; together that sounds slow.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    keep_before_s: float = 0.05,
    keep_after_s: float = 0.12,
    threshold: float = 0.02,
) -> np.ndarray:
    """Cut the silence before and after the voice, keeping a short natural pause.

    Silence is every 10 ms block quieter than ``threshold`` times the loudest block.
    """
    block = max(1, sample_rate // 100)
    usable = len(audio) // block * block
    if usable == 0:  # shorter than one block: nothing to measure
        return audio
    blocks = audio[:usable].astype(np.float64).reshape(-1, block)
    loudness = np.sqrt(np.mean(blocks**2, axis=1))
    voiced = np.flatnonzero(loudness > threshold * loudness.max()) if loudness.max() > 0 else []
    if len(voiced) == 0:
        return audio[:0]
    start = max(0, voiced[0] * block - int(keep_before_s * sample_rate))
    end = min(len(audio), (voiced[-1] + 1) * block + int(keep_after_s * sample_rate))
    return audio[start:end]


def change_speed(audio: np.ndarray, sample_rate: int, rate: float, frame_ms: float = 30.0) -> np.ndarray:
    """Speak ``rate`` times faster without changing the pitch (WSOLA: overlap-add of short pieces, each
    shifted by up to a few milliseconds to where it best continues the previous one)."""
    frame = int(frame_ms / 1000 * sample_rate)
    if rate == 1.0 or len(audio) < 2 * frame:
        return audio
    signal = audio.astype(np.float64)
    hop_out = frame // 2
    hop_in = hop_out * rate
    tolerance = frame // 4  # how far a piece may be shifted to line up with the previous one
    overlap = frame - hop_out
    window = np.hanning(frame)
    frames = int((len(signal) - frame - tolerance) / hop_in)
    output = np.zeros(frames * hop_out + frame)
    weight = np.zeros_like(output)
    previous = 0
    for index in range(frames):
        target = int(index * hop_in)
        if index == 0:
            start = target
        else:
            # The audio that would naturally follow the previous piece; find the shift that matches it best.
            natural = signal[previous + hop_out : previous + hop_out + overlap]
            low, high = max(0, target - tolerance), min(len(signal) - frame, target + tolerance)
            candidates = sliding_window_view(signal[low : high + overlap], overlap)
            start = low + int(np.argmax(candidates @ natural))
        output[index * hop_out : index * hop_out + frame] += signal[start : start + frame] * window
        weight[index * hop_out : index * hop_out + frame] += window
        previous = start
    covered = weight > 1e-3
    output[covered] /= weight[covered]
    output = output[: int(np.flatnonzero(covered)[-1]) + 1]
    return np.clip(output, -32768, 32767).astype(audio.dtype)
