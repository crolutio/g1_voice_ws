"""Check how well the robot hears: spoken test sentences go through the same steps as stt_node.

The sentences are spoken by several Qwen3-TTS voices, at different loudness and with different background noise,
then streamed in 30 ms frames through the voice activity detection, Whisper and the wake word gate. A sentence
counts as heard correctly when exactly the expected request comes out. Run it with the workspace environment:

    python check_speech_recognition.py                       # the settings stt_node uses
    python check_speech_recognition.py --vocabulary-as none  # compare: without the vocabulary

The spoken audio is generated once and cached (about 2 minutes on the RTX 4090 Laptop GPU).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robot_voice_assistant.cuda_libs import preload_cuda_libraries  # noqa: E402
from robot_voice_assistant.speech_recognition import DEFAULT_VOCABULARY, Transcriber  # noqa: E402
from robot_voice_assistant.transcript_filter import is_meaningful_transcript  # noqa: E402
from robot_voice_assistant.vad import EnergyVadSegmenter  # noqa: E402
from robot_voice_assistant.wake_word import DEFAULT_SOUND_ALIKES, WakeWordGate  # noqa: E402

SAMPLE_RATE = 16000
FRAME = 480  # 30 ms
WAKE_WORDS = ("robot", "r1", "r one", "are one")
VOICES = ("ryan", "aiden", "dylan", "eric", "serena", "vivian")
LOUDNESS = (0.25, 0.4, 0.55, 0.7, 0.35, 0.6)  # peak level per voice: quiet to loud speakers
NOISE_LEVELS = {"quiet room": 0.0015, "fan": 0.01, "noisy": 0.025}  # rms of the background noise
CACHE = os.path.expanduser("~/g1_voice_ws/log/stt_check/spoken_sentences_trimmed.npz")


@dataclass(frozen=True)
class Case:
    spoken: tuple[str, ...]  # pieces spoken with a pause in between
    expected: tuple[str, ...]  # accepted requests (Whisper may write numbers as digits)
    starts_conversation: bool = False  # True: spoken with the wake word, the robot is idle before
    pause_s: float = 0.0


CASES = [
    Case(("Robot, what is the capital of India?",), ("what is the capital of India",), True),
    Case(("Robot, tell me about Aideology.",), ("tell me about Aideology",), True),
    Case(("Hey robot, can you wave at me?",), ("can you wave at me",), True),
    Case(("R1, walk forward.",), ("walk forward",), True),
    Case(("Robot, who is the CEO of Aideology?",), ("who is the CEO of Aideology",), True),
    Case(("Who founded Aideology?",), ("who founded Aideology",)),
    Case(("Where is Aideology based?",), ("where is Aideology based",)),
    Case(("What does Aideology do?",), ("what does Aideology do",)),
    Case(("Is Jorge Colotto the CEO?",), ("is Jorge Colotto the CEO",)),
    Case(("Tell me about Dubai Silicon Oasis.",), ("tell me about Dubai Silicon Oasis", "tell me about Dubai's Silicon Oasis")),
    Case(("Turn left.",), ("turn left",)),
    Case(("Turn around.",), ("turn around",)),
    Case(("Walk backward.",), ("walk backward", "walk backwards")),
    Case(("Stop.",), ("stop",)),
    Case(("Keep walking.",), ("keep walking",)),
    Case(("Can you nod your head?",), ("can you nod your head",)),
    Case(("What is twelve times eight?",), ("what is twelve times eight", "what is 12 times 8", "what's 12 times 8")),
    Case(("Tell me a joke.",), ("tell me a joke",)),
    Case(("Goodbye.",), ("goodbye", "good bye")),
    Case(("Stop listening.",), ("stop listening",)),
] + [
    Case(("Robot.", "What is the capital of India?"), ("what is the capital of india",), True, pause)
    for pause in (0.5, 0.9, 1.5)
] + [
    Case(("Robot.", "Tell me about Aideology."), ("tell me about aideology",), True, pause) for pause in (0.5, 0.9, 1.5)
]


CONTRACTIONS = {"where's": "where is", "what's": "what is", "who's": "who is", "that's": "that is"}


def normalize(text: str) -> str:
    words = re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()
    return " ".join(CONTRACTIONS.get(word, word) for word in words)


def word_errors(heard: str, expected: str) -> tuple[int, int]:
    """(edit distance in words, number of expected words)."""
    a, b = normalize(heard).split(), normalize(expected).split()
    row = list(range(len(b) + 1))
    for i, word in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, target in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (word != target))
    return row[len(b)], len(b)


def spoken_audio(tts_device: str) -> dict[str, np.ndarray]:
    """16 kHz float audio per (voice, sentence piece), generated once and cached."""
    pieces = sorted({piece for case in CASES for piece in case.spoken})
    keys = [f"{voice}|{piece}" for voice in VOICES for piece in pieces]
    if os.path.exists(CACHE):
        cached = dict(np.load(CACHE))
        if all(key in cached for key in keys):
            return cached
    import torch

    from robot_voice_assistant.speech_pipeline import Qwen3SpeechEngine

    audio: dict[str, np.ndarray] = {}
    for voice in VOICES:
        engine = Qwen3SpeechEngine(speaker=voice, device=tts_device)
        for number, piece in enumerate(pieces):
            torch.manual_seed(number)  # the voice samples randomly: a fixed seed gives the same test set every time
            speech = engine.synthesize(piece).astype(np.float32) / 32768.0
            loud = np.flatnonzero(np.abs(speech) > 0.02 * np.abs(speech).max())
            speech = speech[max(0, loud[0] - 2400) : loud[-1] + 2400]  # 0.1 s around the voice: pauses are exact
            positions = np.arange(0, len(speech), engine.sample_rate / SAMPLE_RATE)
            audio[f"{voice}|{piece}"] = np.interp(positions, np.arange(len(speech)), speech).astype(np.float32)
        print(f"  generated the sentences in voice {voice}", flush=True)
        del engine
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **audio)
    return audio


def background(samples: int, rms: float, rng: np.random.Generator) -> np.ndarray:
    """Fan-like noise: white noise smoothed towards low frequencies."""
    white = rng.standard_normal(samples)
    smooth = np.convolve(white, np.ones(8) / 8, mode="same")
    return (smooth / (np.sqrt(np.mean(smooth**2)) + 1e-9) * rms).astype(np.float32)


def requests_heard(
    stream: np.ndarray, transcriber: Transcriber, gate: WakeWordGate, end_silence_ms: int, transcripts: list[str]
) -> list[str]:
    """What stt_node would publish for this stream."""
    vad = EnergyVadSegmenter(SAMPLE_RATE, frame_ms=30, threshold_ratio=2.5, min_rms=0.01, end_silence_ms=end_silence_ms)
    requests: list[str] = []
    for index in range(0, len(stream) - FRAME + 1, FRAME):
        now = index / SAMPLE_RATE
        if (greeting := gate.poll(now, speech_in_progress=vad.in_utterance)) is not None:
            requests.append(greeting)
        utterance = vad.push(stream[index : index + FRAME])
        if utterance is None:
            continue
        transcript = transcriber.transcribe(utterance)
        transcripts.append(transcript.text)
        if not is_meaningful_transcript(transcript.text, transcript.no_speech_probability):
            continue
        if (request := gate.accept(transcript.text, now)) is not None:
            requests.append(request)
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--vocabulary-as", default="prompt", choices=("prompt", "hotwords", "none"))
    parser.add_argument("--end-silence-ms", type=int, default=700)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--tts-device", default="cuda:0")
    parser.add_argument("--show-transcripts", action="store_true", help="show what Whisper wrote for mistakes")
    args = parser.parse_args()

    audio = spoken_audio(args.tts_device)
    preload_cuda_libraries()
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    transcriber = Transcriber(model, beam_size=args.beam_size, vocabulary=DEFAULT_VOCABULARY,
                              vocabulary_as=args.vocabulary_as)
    rng = np.random.default_rng(7)
    results: dict[str, list[bool]] = {name: [] for name in NOISE_LEVELS}
    errors = words = invented = invented_commands = 0
    commands = ("walk forward", "walk backward", "turn left", "turn right", "turn around", "keep walking", "stop")
    names = {normalize(word) for name in DEFAULT_VOCABULARY + ("R1", "Unitree") for word in name.split()}
    mistakes: list[str] = []
    started = time.perf_counter()
    for noise_name, noise_rms in NOISE_LEVELS.items():
        for voice, loudness in zip(VOICES, LOUDNESS):
            for case in CASES:
                parts = [np.zeros(SAMPLE_RATE, dtype=np.float32)]  # one second before the speech
                for number, piece in enumerate(case.spoken):
                    speech = audio[f"{voice}|{piece}"]
                    if number:
                        parts.append(np.zeros(int(case.pause_s * SAMPLE_RATE), dtype=np.float32))
                    parts.append(speech / (np.abs(speech).max() + 1e-9) * loudness)
                parts.append(np.zeros(int(4.0 * SAMPLE_RATE), dtype=np.float32))
                stream = np.concatenate(parts)
                stream += background(len(stream), noise_rms, rng)
                gate = WakeWordGate(WAKE_WORDS, sound_alikes=DEFAULT_SOUND_ALIKES)
                if not case.starts_conversation:
                    gate.accept("Robot, let's talk.", now_s=-1.0)  # already in a conversation
                transcripts: list[str] = []
                heard = requests_heard(stream, transcriber, gate, args.end_silence_ms, transcripts)
                correct = len(heard) == 1 and normalize(heard[0]) in {normalize(e) for e in case.expected}
                results[noise_name].append(correct)
                spoken_words = set(normalize(" ".join(case.spoken)).split())
                invented += len([w for w in normalize(" ".join(heard)).split() if w in names and w not in spoken_words])
                spoken_text, heard_text = normalize(" ".join(case.spoken)), normalize(" ".join(heard))
                invented_commands += sum(1 for c in commands if c in heard_text and c not in spoken_text)
                best = min((word_errors(" ".join(heard), e) for e in case.expected), key=lambda pair: pair[0])
                errors, words = errors + best[0], words + best[1]
                if not correct:
                    said = " ... ".join(case.spoken) + (f" (pause {case.pause_s} s)" if case.pause_s else "")
                    whisper = f" (Whisper wrote {transcripts})" if args.show_transcripts else ""
                    mistakes.append(f"[{noise_name}, {voice}] said {said!r}, heard {heard}{whisper}")

    total = [ok for oks in results.values() for ok in oks]
    print(f"\nWhisper {args.model}, vocabulary as {args.vocabulary_as}, end of speech after {args.end_silence_ms} ms "
          f"of silence, beam {args.beam_size} ({time.perf_counter() - started:.0f} s)")
    for noise_name, oks in results.items():
        print(f"  {noise_name:10s} {sum(oks):3d}/{len(oks)} sentences heard exactly right")
    print(f"  {'all':10s} {sum(total):3d}/{len(total)} ({100 * sum(total) / len(total):.1f}%), "
          f"word error rate {100 * errors / max(1, words):.1f}%, invented names {invented}, "
          f"invented commands {invented_commands}")
    for mistake in mistakes:
        print("  -", mistake)


if __name__ == "__main__":
    main()
