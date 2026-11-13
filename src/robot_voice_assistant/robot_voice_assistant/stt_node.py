"""ROS 2 node: listens on the microphone and publishes what the user said."""

from __future__ import annotations

import ctypes
import queue
import shlex
import signal
import subprocess
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, Float64, String

from robot_voice_assistant.cuda_libs import preload_cuda_libraries
from robot_voice_assistant.node_runner import run_node
from robot_voice_assistant.speech_recognition import DEFAULT_VOCABULARY, Transcriber
from robot_voice_assistant.transcript_filter import is_meaningful_transcript
from robot_voice_assistant.vad import EnergyVadSegmenter
from robot_voice_assistant.wake_word import DEFAULT_SOUND_ALIKES, WakeWordGate

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
        self.declare_parameter("languages", ["en"])  # what people may speak; ["en", "ar"]: English and Arabic
        self.declare_parameter("mic_device", "default")
        # Program that records the microphone; it gets arecord's arguments (scripts/fake_microphone.py for tests)
        self.declare_parameter("recorder", "arecord")
        self.declare_parameter("min_rms", 0.01)
        self.declare_parameter("threshold_ratio", 2.5)
        self.declare_parameter("unmute_delay_s", 0.5)
        self.declare_parameter("end_silence_ms", 700)  # this much silence ends a sentence
        # Saying a wake word starts a conversation, which lasts until the person says goodbye (or
        # "stop listening", "that's all", ...). Speech outside a conversation is dropped unlogged.
        self.declare_parameter("wake_words", ["robot", "r1", "r one", "are one"])
        # What Whisper sometimes writes for the wake word ("Robert"); only counts at the start of a sentence
        self.declare_parameter("wake_word_sound_alikes", list(DEFAULT_SOUND_ALIKES))
        self.declare_parameter("conversation_idle_timeout_s", 0.0)  # 0 = a conversation never times out
        self.declare_parameter("wake_word_wait_s", 3.0)  # after the wake word alone, wait this long for the question
        # Names and words Whisper should expect (it is given them as a prompt)
        self.declare_parameter("vocabulary", list(DEFAULT_VOCABULARY))
        self.declare_parameter("beam_size", 5)

        device = str(self.get_parameter("device").value)
        if device == "cuda":
            loaded = preload_cuda_libraries()
            self.get_logger().info(f"Preloaded CUDA libraries: {', '.join(loaded) or 'none'}")

        from faster_whisper import WhisperModel  # imported after the CUDA preload on purpose

        model_size = str(self.get_parameter("model_size").value)
        self.get_logger().info(f"Loading Whisper '{model_size}' on {device}...")
        model = WhisperModel(model_size, device=device, compute_type=str(self.get_parameter("compute_type").value))
        self._transcriber = Transcriber(
            model,
            languages=[str(language) for language in self.get_parameter("languages").value],
            beam_size=int(self.get_parameter("beam_size").value),
            vocabulary=[str(word) for word in self.get_parameter("vocabulary").value if str(word).strip()],
        )
        self._unmute_delay_s = float(self.get_parameter("unmute_delay_s").value)

        self._segmenter = EnergyVadSegmenter(
            sample_rate=SAMPLE_RATE,
            frame_ms=FRAME_MS,
            min_rms=float(self.get_parameter("min_rms").value),
            threshold_ratio=float(self.get_parameter("threshold_ratio").value),
            end_silence_ms=int(self.get_parameter("end_silence_ms").value),
        )
        self._wake_gate = WakeWordGate(
            [str(word) for word in self.get_parameter("wake_words").value],
            idle_timeout_s=float(self.get_parameter("conversation_idle_timeout_s").value),
            wake_word_wait_s=float(self.get_parameter("wake_word_wait_s").value),
            sound_alikes=[str(word) for word in self.get_parameter("wake_word_sound_alikes").value],
        )
        self._in_conversation = False
        self._robot_speaking = False
        self._muted_until = 0.0
        self._utterances: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=4)  # (audio, wall time it ended)

        self._transcript_pub = self.create_publisher(String, "/voice/transcript", 10)
        # Wall-clock time (time.time()) at which the person stopped speaking, published just before each transcript,
        # so the brain and the voice can log how long after that their step finished.
        self._speech_ended_pub = self.create_publisher(Float64, "/voice/speech_ended", 10)
        self._listening_pub = self.create_publisher(Bool, "/voice/listening", 10)
        self._conversation_pub = self.create_publisher(Bool, "/voice/conversation", 10)  # True while in a conversation
        # [rms of the latest 30 ms frame, current speech threshold], ~10 Hz, for tuning min_rms/threshold_ratio.
        self._level_pub = self.create_publisher(Float32MultiArray, "/voice/mic_level", 10)
        self.create_timer(0.1, self._publish_level)
        self.create_timer(30.0, self._log_noise)
        self.create_subscription(Bool, "/voice/speaking", self._on_robot_speaking, 10)
        # the robot greeted someone it saw walk up: they can answer without the wake word
        self.create_subscription(String, "/vision/person_arrived", self._on_person_arrived, 10)

        self._recorder = subprocess.Popen(
            [
                *shlex.split(str(self.get_parameter("recorder").value)),
                "-q", "-D", str(self.get_parameter("mic_device").value),
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
                    self._utterances.put_nowait((utterance, time.time()))
                except queue.Full:
                    self.get_logger().warn("Transcriber is behind; dropping an utterance.")

    def _transcribe_loop(self) -> None:
        while rclpy.ok():
            try:
                utterance, ended_at = self._utterances.get(timeout=0.1)
            except queue.Empty:
                self._greet_if_no_question_followed()
                continue
            if self._is_muted():
                continue
            self._listening_pub.publish(Bool(data=False))
            transcript = self._transcriber.transcribe(utterance)
            self._listening_pub.publish(Bool(data=True))
            if not is_meaningful_transcript(transcript.text, transcript.no_speech_probability):
                continue
            request = self._wake_gate.accept(transcript.text, time.monotonic())
            self._report_conversation_state()
            if request is None:
                if self._wake_gate.in_conversation:
                    self.get_logger().info("Heard the wake word; listening for the question.")
                continue
            wait_s = self._segmenter.end_silence_s
            self.get_logger().info(
                f"Heard ({transcript.language}): {request} ({len(utterance) / SAMPLE_RATE:.1f} s of speech; "
                f"waited {wait_s:.2f} s for silence, speech to text {transcript.seconds:.2f} s)"
            )
            self._speech_ended_pub.publish(Float64(data=ended_at - wait_s))
            self._transcript_pub.publish(String(data=request))

    def _greet_if_no_question_followed(self) -> None:
        greeting = self._wake_gate.poll(time.monotonic(), speech_in_progress=self._segmenter.in_utterance)
        if greeting is not None:
            self.get_logger().info(f"No question followed the wake word; answering {greeting!r}")
            self._transcript_pub.publish(String(data=greeting))

    def _on_person_arrived(self, _msg: String) -> None:
        self._wake_gate.start_conversation(time.monotonic())
        self._report_conversation_state()

    def _report_conversation_state(self) -> None:
        if self._wake_gate.in_conversation == self._in_conversation:
            return
        self._in_conversation = self._wake_gate.in_conversation
        self._conversation_pub.publish(Bool(data=self._in_conversation))
        if self._in_conversation:
            self.get_logger().info("Conversation started: no wake word needed until you say goodbye.")
        else:
            words = ", ".join(f"'{w}'" for w in self.get_parameter("wake_words").value)
            self.get_logger().info(f"Conversation ended. Say a wake word ({words}) to start again.")

    def destroy_node(self) -> None:
        self._recorder.terminate()
        super().destroy_node()


def main() -> None:
    run_node(SttNode)


if __name__ == "__main__":
    main()
