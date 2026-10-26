"""Text-to-speech engines (Qwen3-TTS, Piper) and a pipeline that speaks a reply sentence by sentence.

The next sentence is synthesized while the current one plays, so the robot starts talking after the
first sentence instead of after the whole reply.
"""

from __future__ import annotations

import queue
import subprocess
import threading
from collections.abc import Callable
from typing import Protocol

import numpy as np

from robot_voice_assistant.conversation import split_sentences


class SpeechEngine(Protocol):
    sample_rate: int

    def synthesize(self, text: str) -> np.ndarray:
        """Mono 16-bit audio for ``text``."""
        ...


class AudioPlayer(Protocol):
    def play(self, audio: np.ndarray, sample_rate: int) -> None: ...


class Qwen3SpeechEngine:
    """Qwen3-TTS CustomVoice: preset voices (e.g. "ryan", "aiden") with optional style instructions."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        speaker: str = "ryan",
        language: str = "english",
        instruct: str = "",
        device: str = "cuda:0",
    ) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        self._model = Qwen3TTSModel.from_pretrained(
            model_name, device_map=device, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        self._speaker, self._language, self._instruct = speaker, language, instruct or None
        self.sample_rate = 24000

    def synthesize(self, text: str) -> np.ndarray:
        wavs, self.sample_rate = self._model.generate_custom_voice(
            text=text, speaker=self._speaker, language=self._language, instruct=self._instruct
        )
        return (np.clip(wavs[0], -1.0, 1.0) * 32767).astype(np.int16)


class PiperSpeechEngine:
    """Piper: very fast, lower quality voice."""

    def __init__(self, voice_model: str, use_cuda: bool = False) -> None:
        from piper import PiperVoice

        self._voice = PiperVoice.load(voice_model, use_cuda=use_cuda)
        self.sample_rate = self._voice.config.sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        return np.concatenate([chunk.audio_int16_array for chunk in self._voice.synthesize(text)])


class AplayPlayer:
    def __init__(self, program: str = "aplay") -> None:
        self._program = program

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        subprocess.run(
            [self._program, "-q", "-f", "S16_LE", "-r", str(sample_rate), "-c", "1", "-t", "raw"],
            input=audio.astype(np.int16).tobytes(),
            check=False,
        )


def spoken_chunks(text: str, min_chars: int = 40) -> list[str]:
    """Sentences, with very short ones joined to the next so the voice does not sound choppy."""
    chunks: list[str] = []
    for sentence in split_sentences(text):
        if chunks and len(chunks[-1]) < min_chars:
            chunks[-1] = f"{chunks[-1]} {sentence}"
        else:
            chunks.append(sentence)
    return chunks


_DONE = object()


class SpeechPipeline:
    def __init__(
        self,
        engine: SpeechEngine,
        player: AudioPlayer,
        on_error: Callable[[str], None] = lambda message: None,
    ) -> None:
        self._engine = engine
        self._player = player
        self._on_error = on_error
        self._lock = threading.Lock()  # one reply at a time

    def speak(self, text: str, min_chars: int = 40) -> None:
        """Synthesize and play ``text``; returns when the last sentence has been played."""
        with self._lock:
            ready: queue.Queue[object] = queue.Queue(maxsize=2)

            def synthesize_all() -> None:
                for chunk in spoken_chunks(text, min_chars):
                    try:
                        ready.put(self._engine.synthesize(chunk))
                    except Exception as error:  # one bad sentence must not silence the rest
                        self._on_error(f"Could not synthesize {chunk!r}: {error}")
                ready.put(_DONE)

            producer = threading.Thread(target=synthesize_all, daemon=True)
            producer.start()
            while (audio := ready.get()) is not _DONE:
                self._player.play(audio, self._engine.sample_rate)
            producer.join()
