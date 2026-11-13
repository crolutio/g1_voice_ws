"""ROS 2 node: speaks replies aloud (Qwen3-TTS by default, Piper as a fast alternative).

Publishes /voice/speaking while talking, so the microphone is muted and the robot does not hear itself.
"""

from __future__ import annotations

import os
import queue
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, String

from robot_voice_assistant.language import ARABIC, ENGLISH
from robot_voice_assistant.node_runner import run_node
from robot_voice_assistant.speech_pipeline import (
    AplayPlayer,
    PiperSpeechEngine,
    Qwen3SpeechEngine,
    SpeechEngine,
    FallbackSpeechEngine,
    RivaSpeechEngine,
    SpeechPipeline,
    SpeechTiming,
)
from robot_voice_assistant.timing import seconds_since

DEFAULT_PIPER_VOICE = os.path.expanduser("~/g1_voice_ws/models/en_US-lessac-medium.onnx")


class TtsNode(Node):
    def __init__(self) -> None:
        super().__init__("tts_node")
        # riva: NVIDIA Riva voices in NVIDIA's cloud (needs NVIDIA_API_KEY), with Qwen3-TTS when unreachable;
        # qwen3: Qwen3-TTS on this computer; piper: very fast local voice
        self.declare_parameter("tts_engine", "riva")
        self.declare_parameter("riva_voice", "Magpie-Multilingual.EN-US.Ray")
        self.declare_parameter("riva_language", "en-US")
        # Arabic replies are spoken with this voice (the local voices speak no Arabic)
        self.declare_parameter("riva_voice_arabic", "Magpie-Multilingual.AR-XA.Ray")
        self.declare_parameter("riva_language_arabic", "ar-XA")
        self.declare_parameter("riva_sample_rate", 44100)
        self.declare_parameter("riva_speaking_rate", 0.9)  # Riva voices speak fast (about 200 words/min)
        self.declare_parameter("qwen_model", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
        self.declare_parameter("speaker", "ryan")
        self.declare_parameter("language", "english")
        self.declare_parameter("instruct", "")  # e.g. "Speak in a warm, friendly tone."
        self.declare_parameter("fast_decoding", True)  # Qwen3-TTS: about twice as fast; false = library decoding
        self.declare_parameter("voice_model", DEFAULT_PIPER_VOICE)  # Piper voice file
        self.declare_parameter("use_cuda", False)  # Piper only
        self.declare_parameter("player", "aplay")
        # 1.0 = the voice's own pace (about 150 words per minute once the silence is trimmed); 1.1 = normal
        # conversation (about 165); the pitch stays the same
        self.declare_parameter("speaking_rate", 1.1)

        engine, speaking_rate = self._load_engine()
        self._pipeline = SpeechPipeline(
            engine,
            AplayPlayer(str(self.get_parameter("player").value)),
            on_error=self.get_logger().error,
            speaking_rate=speaking_rate,
        )
        # (reply text, wall time it arrived, wall time the person stopped speaking or None)
        self._replies: queue.Queue[tuple[str, float, float | None]] = queue.Queue()
        self._speech_ended: float | None = None
        self._speaking_pub = self.create_publisher(Bool, "/voice/speaking", 10)
        self.create_subscription(Float64, "/voice/speech_ended", self._on_speech_ended, 10)
        self.create_subscription(String, "/voice/reply", self._on_reply, 10)
        threading.Thread(target=self._speak_loop, daemon=True).start()

    def _load_engine(self) -> tuple[SpeechEngine, float]:
        """The voice engine and the speaking rate to play it at."""
        choice = str(self.get_parameter("tts_engine").value)
        qwen_rate = float(self.get_parameter("speaking_rate").value)
        if choice == "riva":
            api_key = os.environ.get("NVIDIA_API_KEY", "")
            if not api_key:
                self.get_logger().warn("NVIDIA_API_KEY is not set, so the Riva voice is off; using Qwen3-TTS.")
            else:
                riva_voice = str(self.get_parameter("riva_voice").value)
                riva_language = str(self.get_parameter("riva_language").value)
                arabic_voice = str(self.get_parameter("riva_voice_arabic").value)
                riva = RivaSpeechEngine(
                    api_key,
                    voice=riva_voice,
                    language=riva_language,
                    sample_rate=int(self.get_parameter("riva_sample_rate").value),
                    voices={ENGLISH: (riva_voice, riva_language),
                            ARABIC: (arabic_voice, str(self.get_parameter("riva_language_arabic").value))},
                )
                riva_rate = float(self.get_parameter("riva_speaking_rate").value)
                local, _ = self._load_local_engine("qwen3", qwen_rate, ready_label="Fallback voice ready")
                engine = FallbackSpeechEngine(
                    riva,
                    local,
                    fallback_speed=qwen_rate / riva_rate,
                    on_fallback=lambda reason: self.get_logger().warn(
                        f"Riva voice unavailable ({reason}); speaking with the local voice for 30 s."
                    ),
                )
                self._warm_up(riva)
                self.get_logger().info(f"TTS ready (NVIDIA Riva, voices {riva_voice} and {arabic_voice}; "
                                       "local Qwen3-TTS, English only, if unreachable).")
                return engine, riva_rate
        return self._load_local_engine("qwen3" if choice == "riva" else choice, qwen_rate)

    def _warm_up(self, riva: RivaSpeechEngine) -> None:
        """Open the connection to NVIDIA's cloud now: the first request takes seconds longer than the rest."""
        started = time.monotonic()
        try:
            riva.synthesize("Hello.")
            self.get_logger().info(f"Riva voice warmed up in {time.monotonic() - started:.1f} s.")
        except Exception as error:
            self.get_logger().warn(f"Riva voice warm-up failed ({type(error).__name__}: {error}); it will retry when speaking.")

    def _load_local_engine(
        self, choice: str, qwen_rate: float, ready_label: str = "TTS ready"
    ) -> tuple[SpeechEngine, float]:
        if choice == "qwen3":
            try:
                engine = Qwen3SpeechEngine(
                    str(self.get_parameter("qwen_model").value),
                    speaker=str(self.get_parameter("speaker").value),
                    language=str(self.get_parameter("language").value),
                    instruct=str(self.get_parameter("instruct").value),
                    fast_decoding=bool(self.get_parameter("fast_decoding").value),
                    on_warning=self.get_logger().warn,
                )
                decoding = ", fast decoding" if engine.fast_decoding else ""
                self.get_logger().info(f"{ready_label} (Qwen3-TTS, voice {self.get_parameter('speaker').value}{decoding}).")
                return engine, qwen_rate
            except Exception as error:  # e.g. not enough GPU memory
                self.get_logger().error(f"Could not load Qwen3-TTS ({error}); using Piper instead.")
        engine = PiperSpeechEngine(str(self.get_parameter("voice_model").value), bool(self.get_parameter("use_cuda").value))
        self.get_logger().info(f"{ready_label} (Piper, {os.path.basename(str(self.get_parameter('voice_model').value))}).")
        return engine, 1.0

    def _on_speech_ended(self, msg: Float64) -> None:
        self._speech_ended = msg.data

    def _on_reply(self, msg: String) -> None:
        speech_ended, self._speech_ended = self._speech_ended, None
        if msg.data.strip():
            self._replies.put((msg.data.strip(), time.time(), speech_ended))

    def _log_first_audio(self, seconds: float, queued_s: float, speech_ended: float | None) -> None:
        line = f"Voice started {queued_s + seconds:.2f} s after the reply arrived"
        if queued_s > 0.05:
            line += f" (of which {queued_s:.2f} s waiting for the previous reply to finish)"
        if (since_speech := seconds_since(speech_ended)) is not None:
            line += f"; {since_speech:.2f} s after you stopped speaking"
        self.get_logger().info(line)

    def _log_finished(self, timing: SpeechTiming) -> None:
        pieces = ", ".join(f"{seconds:.2f} s ({chars} characters)" for chars, seconds in timing.pieces)
        self.get_logger().info(f"Finished speaking after {timing.total_s:.2f} s; voice generated in {pieces or 'nothing'}")

    def _speak_loop(self) -> None:
        while rclpy.ok():
            text, arrived, speech_ended = self._replies.get()
            queued_s = time.time() - arrived
            self._speaking_pub.publish(Bool(data=True))
            try:
                timing = self._pipeline.speak(
                    text,
                    on_first_audio=lambda seconds: self._log_first_audio(seconds, queued_s, speech_ended),
                )
                self._log_finished(timing)
            except Exception as error:  # audio device problems must not kill the node
                self.get_logger().error(f"Playback failed: {error}")
            finally:
                self._speaking_pub.publish(Bool(data=False))


def main() -> None:
    run_node(TtsNode)


if __name__ == "__main__":
    main()
