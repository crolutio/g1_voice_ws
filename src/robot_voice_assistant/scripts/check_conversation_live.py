"""Live test of whole conversations: speaks into the running assistant through scripts/fake_microphone.py.

Start the assistant with the fake microphone (see "Test whole conversations" in README.md), then run:

    python check_conversation_live.py --spool /tmp/robot_mic

Every sentence is spoken by Qwen3-TTS and played into stt_node like a microphone would hear it. The script
waits for the robot to finish each reply, like a person, and checks what was heard, the reply, the gesture,
the walking motion and whether the conversation is still on.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SAMPLE_RATE = 16000
CACHE = os.path.expanduser("~/g1_voice_ws/log/stt_check/conversations.npz")


@dataclass(frozen=True)
class Step:
    said: tuple[str, ...]  # pieces, spoken with ``pause_s`` in between
    answered: bool = True
    reply_mentions: tuple[str, ...] = ()  # any of these (lower case) must be in the reply
    motion: str | None = None
    gesture_prefix: str | None = None
    in_conversation_after: bool = True
    pause_s: float = 0.0


@dataclass(frozen=True)
class Scenario:
    voice: str
    loudness: float
    steps: tuple[Step, ...]


SCENARIOS = {
    "quiet room, voice ryan": Scenario("ryan", 0.5, (
        Step(("Robot, what is the capital of India?",), reply_mentions=("delhi",)),
        Step(("And what about France?",), reply_mentions=("paris",)),
        Step(("Tell me about Aideology.",), reply_mentions=("dubai", "ai ")),
        Step(("Walk forward.",), motion="walk_forward"),
        Step(("Stop.",), motion="stop"),
        Step(("Who is the CEO of Aideology?",), reply_mentions=("colotto",)),
        Step(("Goodbye.",), in_conversation_after=False),
        Step(("What time is it?",), answered=False, in_conversation_after=False),
        Step(("Robot.", "Turn left."), motion="turn_left", pause_s=1.0),
        Step(("That's all, thank you.",), in_conversation_after=False),
    )),
    "fan noise, voice serena": Scenario("serena", 0.45, (
        Step(("Robot.",), reply_mentions=("hello", "hi", "hey")),
        Step(("Can you wave at me?",), gesture_prefix="wave"),
        Step(("What is twelve times eight?",), reply_mentions=("96", "ninety-six", "ninety six")),
        Step(("Turn around.",), motion="turn_around"),
        Step(("Stop listening.",), in_conversation_after=False),
        Step(("Tell me a joke.",), answered=False, in_conversation_after=False),
    )),
    "louder noise, voice eric": Scenario("eric", 0.55, (
        Step(("Hey robot, where is Aideology based?",), reply_mentions=("dubai",)),
        Step(("Walk backward.",), motion="walk_backward"),
        Step(("Keep walking.",), motion="keep_walking"),
        Step(("Stop.",), motion="stop"),
        Step(("Go to sleep.",), in_conversation_after=False),
    )),
}


@dataclass
class Observed:
    heard: list[str] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)
    gestures: list[str] = field(default_factory=list)
    motions: list[str] = field(default_factory=list)
    speaking_finished: int = 0


class Listener(Node):
    def __init__(self) -> None:
        super().__init__("conversation_check")
        self.observed = Observed()
        self.in_conversation = False
        self.create_subscription(String, "/voice/transcript", lambda m: self.observed.heard.append(m.data), 10)
        self.create_subscription(String, "/voice/reply", lambda m: self.observed.replies.append(m.data), 10)
        self.create_subscription(String, "/robot/gesture", lambda m: self.observed.gestures.append(m.data), 10)
        self.create_subscription(String, "/robot/motion", lambda m: self.observed.motions.append(m.data), 10)
        self.create_subscription(Bool, "/voice/speaking", self._on_speaking, 10)
        self.create_subscription(Bool, "/voice/conversation", self._on_conversation, 10)

    def _on_speaking(self, msg: Bool) -> None:
        if not msg.data:
            self.observed.speaking_finished += 1

    def _on_conversation(self, msg: Bool) -> None:
        self.in_conversation = msg.data

    def spin_for(self, seconds: float, until: object = None) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end and not (until and until()):  # type: ignore[operator]
            rclpy.spin_once(self, timeout_sec=0.05)


def spoken(engine: object, text: str, seed: int) -> np.ndarray:
    import torch

    torch.manual_seed(seed)
    speech = engine.synthesize(text).astype(np.float32) / 32768.0  # type: ignore[attr-defined]
    loud = np.flatnonzero(np.abs(speech) > 0.02 * np.abs(speech).max())
    speech = speech[max(0, loud[0] - 2400) : loud[-1] + 2400]
    positions = np.arange(0, len(speech), engine.sample_rate / SAMPLE_RATE)  # type: ignore[attr-defined]
    return np.interp(positions, np.arange(len(speech)), speech).astype(np.float32)


def prepared_audio() -> dict[str, np.ndarray]:
    """Every spoken step, generated once (on the GPU) and cached, so the test itself needs no GPU."""
    keys = {f"{name}|{number}": (scenario, step, number)
            for name, scenario in SCENARIOS.items() for number, step in enumerate(scenario.steps, 1)}
    if os.path.exists(CACHE):
        cached = dict(np.load(CACHE))
        if set(keys) <= set(cached):
            return cached
    from robot_voice_assistant.speech_pipeline import Qwen3SpeechEngine

    audio: dict[str, np.ndarray] = {}
    for voice in sorted({scenario.voice for scenario in SCENARIOS.values()}):
        engine = Qwen3SpeechEngine(speaker=voice)
        for key, (scenario, step, number) in keys.items():
            if scenario.voice != voice:
                continue
            pieces = [spoken(engine, piece, seed=number * 10 + index) for index, piece in enumerate(step.said)]
            silence = np.zeros(int(step.pause_s * SAMPLE_RATE), dtype=np.float32)
            joined = np.concatenate([part for piece in pieces for part in (piece, silence)][:-1])
            audio[key] = joined / (np.abs(joined).max() + 1e-9) * scenario.loudness
        del engine
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **audio)
    return audio


def check(step: Step, observed: Observed, in_conversation: bool) -> list[str]:
    problems = []
    if step.answered and not observed.replies:
        problems.append("no reply")
    if not step.answered and (observed.heard or observed.replies):
        problems.append("answered speech it should have ignored")
    reply = " ".join(observed.replies).lower()
    if step.reply_mentions and not any(word in reply for word in step.reply_mentions):
        problems.append(f"reply does not mention {' or '.join(step.reply_mentions)}")
    if step.motion and step.motion not in observed.motions:
        problems.append(f"no {step.motion} motion")
    if step.gesture_prefix and not any(g.startswith(step.gesture_prefix) for g in observed.gestures):
        problems.append(f"no {step.gesture_prefix} gesture")
    if in_conversation != step.in_conversation_after:
        problems.append("conversation " + ("still on" if in_conversation else "ended"))
    if len(observed.heard) > 1:
        problems.append(f"heard as {len(observed.heard)} requests")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spool", help="the fake microphone's spool folder")
    parser.add_argument("--prepare", action="store_true", help="only generate the spoken audio (before starting)")
    parser.add_argument("--scenario", action="append", choices=list(SCENARIOS), help="default: all")
    args = parser.parse_args()

    audio_for = prepared_audio()
    if args.prepare:
        print(f"spoken audio ready in {CACHE}")
        return
    if not args.spool:
        parser.error("--spool is needed")
    rclpy.init()
    listener = Listener()
    listener.spin_for(2.0)
    passed = total = 0
    for name in args.scenario or list(SCENARIOS):
        scenario = SCENARIOS[name]
        print(f"\n=== {name} ===", flush=True)
        for number, step in enumerate(scenario.steps, 1):
            audio = audio_for[f"{name}|{number}"]
            listener.observed = Observed()
            path = os.path.join(args.spool, f"{time.time():.3f}.raw")
            (audio * 32767).astype("<i2").tofile(path + ".part")
            os.replace(path + ".part", path)
            started = time.monotonic()
            if step.answered:
                listener.spin_for(45.0, until=lambda: listener.observed.speaking_finished > 0)
                listener.spin_for(1.5)  # the robot unmutes the microphone, the person takes a breath
            else:
                listener.spin_for(len(audio) / SAMPLE_RATE + 6.0)
            observed = listener.observed
            problems = check(step, observed, listener.in_conversation)
            total += 1
            passed += not problems
            said = " ... ".join(step.said)
            print(f"{number:2d}. said:  {said!r}" + (f" (pause {step.pause_s} s)" if step.pause_s else ""))
            print(f"    heard: {observed.heard or '(nothing)'}")
            print(f"    reply: {observed.replies or '(none)'}  gesture={observed.gestures} motion={observed.motions}")
            print(f"    conversation {'on' if listener.in_conversation else 'off'}, {time.monotonic() - started:.1f} s"
                  f"  ->  {'PASS' if not problems else 'FAIL: ' + '; '.join(problems)}", flush=True)
    print(f"\n{passed}/{total} steps passed")
    listener.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
