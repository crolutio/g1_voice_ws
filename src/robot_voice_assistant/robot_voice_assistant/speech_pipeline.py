"""Text-to-speech engines (Qwen3-TTS, Piper) and a pipeline that speaks a reply sentence by sentence.

The next sentence is synthesized while the current one plays, so the robot starts talking after the
first sentence instead of after the whole reply.
"""

from __future__ import annotations

import queue
import re
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
        fast_decoding: bool = True,
        on_warning: Callable[[str], None] = lambda message: None,
    ) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        self._model = Qwen3TTSModel.from_pretrained(
            model_name, device_map=device, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        self._speaker, self._language, self._instruct = speaker, language, instruct or None
        self.sample_rate = 24000
        self.fast_decoding = False
        if fast_decoding:
            from robot_voice_assistant.fast_code_predictor import FastCodePredictor

            code_predictor = self._model.model.talker.code_predictor
            library_generate = code_predictor.generate
            try:
                FastCodePredictor(code_predictor).install()
                self.synthesize("Hello.")  # captures the CUDA graph now rather than during the first reply
                self.fast_decoding = True
            except Exception as error:  # an unexpected library version must not cost the voice
                code_predictor.generate = library_generate
                on_warning(f"Fast Qwen3-TTS decoding unavailable ({error}); using the slower library decoding.")
        self.synthesize("Hello.")  # first-use warm-up of the GPU kernels

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


# A natural place to pause inside a sentence: after a comma, semicolon or colon, or before a joining word
_CLAUSE_BREAK = re.compile(r"(?<=[,;:])\s+|\s+(?=(?:and|but|so|because|which|that|who|where|while|using)\s)")


def _split_at_clauses(sentence: str, first_limit: int, limit: int, min_piece: int = 20) -> list[str]:
    """Cut a long sentence at clause breaks: the first piece about ``first_limit`` characters, the rest ``limit``.

    The last break within the limit is used; when there is none, the nearest one past it (up to ``limit``).
    """
    pieces: list[str] = []
    while len(sentence) > (allowed := first_limit if not pieces else limit):
        breaks = [
            match
            for match in _CLAUSE_BREAK.finditer(sentence)
            if min_piece <= match.start() <= limit and len(sentence) - match.end() >= min_piece
        ]
        within = [match for match in breaks if match.start() <= allowed]
        if not breaks:
            break
        cut = within[-1] if within else breaks[0]
        pieces.append(sentence[: cut.start()].strip())
        sentence = sentence[cut.end() :].strip()
    return [*pieces, sentence]


def spoken_chunks(
    text: str, min_chars: int = 40, max_chars: int = 100, first_max_chars: int = 60, first_min_chars: int = 12
) -> list[str]:
    """The pieces of a reply that are synthesized one at a time.

    The first piece is kept short so the robot starts talking soon; the next pieces are synthesized while it
    plays. Very short sentences are joined to the next one so the voice does not sound choppy, and long
    sentences are cut at a comma or a joining word.
    """
    joined: list[str] = []
    for sentence in split_sentences(text):
        shortest = first_min_chars if len(joined) == 1 else min_chars
        if joined and len(joined[-1]) < shortest:
            joined[-1] = f"{joined[-1]} {sentence}"
        else:
            joined.append(sentence)
    chunks: list[str] = []
    for sentence in joined:
        chunks += _split_at_clauses(sentence, first_max_chars if not chunks else max_chars, max_chars)
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
