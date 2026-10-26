"""ROS 2 node: listens on the microphone and publishes what the user said."""

from __future__ import annotations

import ctypes
import queue
import signal
import subprocess
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, String

from robot_voice_assistant.cuda_libs import preload_cuda_libraries
from robot_voice_assistant.node_runner import run_node
from robot_voice_assistant.transcript_filter import is_meaningful_transcript
from robot_voice_assistant.vad import EnergyVadSegmenter
from robot_voice_assistant.wake_word import WakeWordGate

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
BYTES_PER_SAMPLE = 2


def _die_with_parent() -> None:
    """Make arecord exit if this node dies, so the microphone is never left open."""
    pr_set_pdeathsig = 1
    ctypes.CDLL("libc.so.6").prctl(pr_set_pdeathsig, signal.SIGTERM)


class SttNode(Node):
    def __init__(self) -> None:
        super().__init__("stt_node")
        self.declare_parameter("model_size", "small.en")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("compute_type", "float16")
        self.declare_parameter("language", "en")
        self.declare_parameter("mic_device", "default")
        self.declare_parameter("min_rms", 0.01)
        self.declare_parameter("threshold_ratio", 2.5)
        self.declare_parameter("unmute_delay_s", 0.5)
        # Only speech addressed to the robot is answered; other conversations are dropped unlogged.
        self.declare_parameter("wake_words", ["robot", "r1", "r one", "are one"])
        self.declare_parameter("follow_up_s", 8.0)

        device = str(self.get_parameter("device").value)
        if device == "cuda":
            loaded = preload_cuda_libraries()
            self.get_logger().info(f"Preloaded CUDA libraries: {', '.join(loaded) or 'none'}")

        from faster_whisper import WhisperModel  # imported after the CUDA preload on purpose

        model_size = str(self.get_parameter("model_size").value)
        self.get_logger().info(f"Loading Whisper '{model_size}' on {device}...")
        self._model = WhisperModel(model_size, device=device, compute_type=str(self.get_parameter("compute_type").value))
        self._language = str(self.get_parameter("language").value)
        self._unmute_delay_s = float(self.get_parameter("unmute_delay_s").value)

        self._segmenter = EnergyVadSegmenter(
            sample_rate=SAMPLE_RATE,
            frame_ms=FRAME_MS,
            min_rms=float(self.get_parameter("min_rms").value),
            threshold_ratio=float(self.get_parameter("threshold_ratio").value),
        )
        self._wake_gate = WakeWordGate(
            [str(word) for word in self.get_parameter("wake_words").value],
            float(self.get_parameter("follow_up_s").value),
        )
        self._robot_speaking = False
        self._muted_until = 0.0
        self._utterances: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)

        self._transcript_pub = self.create_publisher(String, "/voice/transcript", 10)
        self._listening_pub = self.create_publisher(Bool, "/voice/listening", 10)
        # [rms of the latest 30 ms frame, current speech threshold], ~10 Hz, for tuning min_rms/threshold_ratio.
        self._level_pub = self.create_publisher(Float32MultiArray, "/voice/mic_level", 10)
        self.create_timer(0.1, self._publish_level)
        self.create_timer(30.0, self._log_noise)
        self.create_subscription(Bool, "/voice/speaking", self._on_robot_speaking, 10)

        self._recorder = subprocess.Popen(
            [
                "arecord", "-q", "-D", str(self.get_parameter("mic_device").value),
                "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw",
            ],
            stdout=subprocess.PIPE,
            preexec_fn=_die_with_parent,
        )
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._transcribe_loop, daemon=True).start()
        words = ", ".join(f"'{w}'" for w in self.get_parameter("wake_words").value)
        self.get_logger().info(f"Listening. Say a wake word ({words}) to talk to the robot.")

    def _publish_level(self) -> None:
        self._level_pub.publish(Float32MultiArray(data=[self._segmenter.last_rms, self._segmenter.threshold]))

    def _log_noise(self) -> None:
        self.get_logger().info(
            f"Mic noise floor {self._segmenter.noise_floor:.4f}, speech threshold {self._segmenter.threshold:.4f}"
        )

    def _on_robot_speaking(self, msg: Bool) -> None:
        self._robot_speaking = msg.data
        if msg.data:
            self._segmenter.reset()
        else:
            self._muted_until = time.monotonic() + self._unmute_delay_s
            self._wake_gate.note_reply_finished(time.monotonic())

    def _is_muted(self) -> bool:
        return self._robot_speaking or time.monotonic() < self._muted_until

    def _capture_loop(self) -> None:
        assert self._recorder.stdout is not None
        frame_bytes = FRAME_SAMPLES * BYTES_PER_SAMPLE
        while rclpy.ok():
            data = self._recorder.stdout.read(frame_bytes)
            if len(data) < frame_bytes:
                self.get_logger().error("Microphone stream ended (is arecord able to open the device?).")
                return
            if self._is_muted():
                continue
            frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            utterance = self._segmenter.push(frame)
            if utterance is not None:
                try:
                    self._utterances.put_nowait(utterance)
                except queue.Full:
                    self.get_logger().warn("Transcriber is behind; dropping an utterance.")

    def _transcribe_loop(self) -> None:
        while rclpy.ok():
            utterance = self._utterances.get()
            if self._is_muted():
                continue
            self._listening_pub.publish(Bool(data=False))
            segments, _ = self._model.transcribe(utterance, language=self._language, beam_size=5)
            segment_list = list(segments)
            text = "".join(segment.text for segment in segment_list).strip()
            no_speech = max((segment.no_speech_prob for segment in segment_list), default=1.0)
            self._listening_pub.publish(Bool(data=True))
            if not is_meaningful_transcript(text, no_speech):
                continue
            request = self._wake_gate.accept(text, time.monotonic())
            if request is None:
                self.get_logger().debug("Ignored speech that was not addressed to the robot.")
                continue
            self.get_logger().info(f"Heard: {request}")
            self._transcript_pub.publish(String(data=request))

    def destroy_node(self) -> None:
        self._recorder.terminate()
        super().destroy_node()


def main() -> None:
    run_node(SttNode)


if __name__ == "__main__":
    main()
