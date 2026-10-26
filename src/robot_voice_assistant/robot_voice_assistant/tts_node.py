"""ROS 2 node: speaks replies aloud (Qwen3-TTS by default, Piper as a fast alternative).

Publishes /voice/speaking while talking, so the microphone is muted and the robot does not hear itself.
"""

from __future__ import annotations

import os
import queue
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from robot_voice_assistant.node_runner import run_node
from robot_voice_assistant.speech_pipeline import (
    AplayPlayer,
    PiperSpeechEngine,
    Qwen3SpeechEngine,
    SpeechEngine,
    SpeechPipeline,
)

DEFAULT_PIPER_VOICE = os.path.expanduser("~/g1_voice_ws/models/en_US-lessac-medium.onnx")


class TtsNode(Node):
    def __init__(self) -> None:
        super().__init__("tts_node")
        self.declare_parameter("tts_engine", "qwen3")  # qwen3 or piper
        self.declare_parameter("qwen_model", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
        self.declare_parameter("speaker", "ryan")
        self.declare_parameter("language", "english")
        self.declare_parameter("instruct", "")  # e.g. "Speak in a warm, friendly tone."
        self.declare_parameter("fast_decoding", True)  # Qwen3-TTS: about twice as fast; false = library decoding
        self.declare_parameter("voice_model", DEFAULT_PIPER_VOICE)  # Piper voice file
        self.declare_parameter("use_cuda", False)  # Piper only
        self.declare_parameter("player", "aplay")

        engine = self._load_engine()
        self._pipeline = SpeechPipeline(
            engine, AplayPlayer(str(self.get_parameter("player").value)), on_error=self.get_logger().error
        )
        self._replies: queue.Queue[str] = queue.Queue()
        self._speaking_pub = self.create_publisher(Bool, "/voice/speaking", 10)
        self.create_subscription(String, "/voice/reply", self._on_reply, 10)
        threading.Thread(target=self._speak_loop, daemon=True).start()

    def _load_engine(self) -> SpeechEngine:
        if self.get_parameter("tts_engine").value == "qwen3":
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
                self.get_logger().info(f"TTS ready (Qwen3-TTS, voice {self.get_parameter('speaker').value}{decoding}).")
                return engine
            except Exception as error:  # e.g. not enough GPU memory
                self.get_logger().error(f"Could not load Qwen3-TTS ({error}); using Piper instead.")
        engine = PiperSpeechEngine(str(self.get_parameter("voice_model").value), bool(self.get_parameter("use_cuda").value))
        self.get_logger().info(f"TTS ready (Piper, {os.path.basename(str(self.get_parameter('voice_model').value))}).")
        return engine

    def _on_reply(self, msg: String) -> None:
        if msg.data.strip():
            self._replies.put(msg.data.strip())

    def _speak_loop(self) -> None:
        while rclpy.ok():
            text = self._replies.get()
            self._speaking_pub.publish(Bool(data=True))
            try:
                self._pipeline.speak(text)
            except Exception as error:  # audio device problems must not kill the node
                self.get_logger().error(f"Playback failed: {error}")
            finally:
                self._speaking_pub.publish(Bool(data=False))


def main() -> None:
    run_node(TtsNode)


if __name__ == "__main__":
    main()
