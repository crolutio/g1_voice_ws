"""ROS 2 node: turns transcripts into a spoken reply, a gesture and (when walking) a motion.

Language model: Groq (qwen/qwen3.8-27b by default, key from the GROQ_API_KEY environment variable), falling
back to a local Ollama model. Qwen3-Embedding finds relevant knowledge and memories for every question.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict

from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Float64, String

from robot_voice_assistant.assistant_factory import BrainSettings, create_backend, create_conversation
from robot_voice_assistant.conversation import Conversation
from robot_voice_assistant.language import ARABIC, ENGLISH, language_of
from robot_voice_assistant.motions import is_stop_request
from robot_voice_assistant.node_runner import run_node
from robot_voice_assistant.ollama_client import OllamaClient
from robot_voice_assistant.scene_context import in_conversation
from robot_voice_assistant.timing import seconds_since

GREETING_FOR_PERSON = "Hello, may I help you?"
# Said when the motion node stops or refuses a motion to keep its distance from someone.
SAFE_DISTANCE_REPLY = "I'll stop here, to keep a safe distance from you."
SAFE_DISTANCE_REPLY_TURN = "You're standing quite close, so I won't turn right now."
NO_ANSWER_REPLY = "Sorry, I can't think right now. Please try again in a moment."
# The same, said to someone speaking Arabic (the greeting stays English: the visitor has not spoken yet)
ARABIC_REPLIES = {
    SAFE_DISTANCE_REPLY: "سأتوقف هنا لأحافظ على مسافة آمنة منك.",
    SAFE_DISTANCE_REPLY_TURN: "أنت قريب مني جدا، لذلك لن أستدير الآن.",
    NO_ANSWER_REPLY: "عذرا، لا أستطيع التفكير الآن. حاول مرة أخرى بعد قليل.",
}
SCENE_STALE_S = 3.0  # an older camera description no longer shows what is in front of the robot
DEFAULT_KNOWLEDGE_DIR = os.path.join(get_package_share_directory("robot_voice_assistant"), "config", "knowledge")


class BrainNode(Node):
    def __init__(self) -> None:
        super().__init__("brain_node")
        defaults = BrainSettings()
        for name, value in vars(defaults).items():
            self.declare_parameter(name, value)
        settings = BrainSettings(**{name: self.get_parameter(name).value for name in vars(defaults)})
        settings.knowledge_dir = os.path.expanduser(settings.knowledge_dir or DEFAULT_KNOWLEDGE_DIR)
        settings.memory_file = os.path.expanduser(settings.memory_file)
        self._settings = settings

        self._backend, description = create_backend(settings)
        self.get_logger().info(f"Language model: {description}")
        self._conversation: Conversation | None = None
        self._busy = threading.Lock()
        self._speech_ended: float | None = None  # when the person stopped saying the current request (wall clock)
        self._language = ENGLISH  # what the person spoke last; the robot's fixed sentences are said in it
        # What was said while the brain was busy, answered next (several pieces are joined)
        self._pending: tuple[str, float | None] | None = None
        self._pending_lock = threading.Lock()

        self._reply_pub = self.create_publisher(String, "/voice/reply", 10)
        self._gesture_pub = self.create_publisher(String, "/robot/gesture", 10)
        self._motion_pub = self.create_publisher(String, "/robot/motion", 10)
        self._usage_pub = self.create_publisher(String, "/voice/usage", 10)
        self._last_answer_at: float | None = None  # tokens and timing per answer, as JSON
        self.create_subscription(Float64, "/voice/speech_ended", self._on_speech_ended, 10)
        self.create_subscription(String, "/voice/transcript", self._on_transcript, 10)
        self.create_subscription(String, "/voice/reset", self._on_reset, 10)
        self.create_subscription(String, "/voice/forget", self._on_forget, 10)
        # what the head camera sees (vision_node); None until the vision node runs
        self._scene: str | None = None
        self._scene_time = 0.0
        self.create_subscription(String, "/vision/scene", self._on_scene, 10)
        self.create_subscription(String, "/vision/person_arrived", self._on_person_arrived, 10)
        self.create_subscription(String, "/robot/motion_blocked", self._on_motion_blocked, 10)
        threading.Thread(target=self._start, daemon=True).start()

    def _start(self) -> None:
        """Load the embedding model and the knowledge base (takes a little while), then start answering."""
        with self._busy:
            try:
                self._conversation, notes = create_conversation(self._settings, self._backend, self._surroundings)
            except Exception as error:  # e.g. no GPU memory for the embedding model
                self.get_logger().error(f"Could not load embeddings ({error}); continuing without them.")
                self._settings.use_embeddings = False
                self._conversation, notes = create_conversation(self._settings, self._backend, self._surroundings)
            for note in notes:
                self.get_logger().info(note)
            if self._backend.has_fallback:  # loading it now keeps the first fallback answer fast
                try:
                    OllamaClient(self._settings.ollama_url, self._settings.fallback_model).warm_up()
                except Exception as error:
                    self.get_logger().warn(f"Local fallback model not available ({error}). Is Ollama running?")
        walking = "walking enabled" if self._settings.can_walk else "standing in place"
        self.get_logger().info(f"Brain ready ({walking}).")
        with self._pending_lock:  # something said while loading
            pending, self._pending = self._pending, None
            if pending is not None and self._busy.acquire(blocking=False):
                threading.Thread(target=self._respond, args=pending, daemon=True).start()

    def _on_reset(self, _msg: String) -> None:
        with self._busy:
            if self._conversation is not None:
                self._conversation.reset()
            self._last_answer_at = None  # the next person is a new visitor
            self._language = ENGLISH
        self.get_logger().info("Conversation history cleared.")

    def _on_forget(self, _msg: String) -> None:
        """Delete everything the robot remembers from earlier conversations."""
        with self._busy:
            if self._conversation is not None:
                self._conversation.forget_memories()
        self.get_logger().info("Long-term memory deleted.")

    def _on_scene(self, msg: String) -> None:
        self._scene, self._scene_time = msg.data, time.monotonic()

    def _surroundings(self) -> str | None:
        """What the camera sees now, for the prompt; None without a camera, "unclear" if the view is out of date."""
        if self._scene is None:
            return None
        return self._scene if time.monotonic() - self._scene_time < SCENE_STALE_S else "the camera view is not available"

    def _on_motion_blocked(self, msg: String) -> None:
        """The motion node kept the robot away from someone: say so, so the person knows why it stopped."""
        reply = self._in_their_language(SAFE_DISTANCE_REPLY_TURN if "reach" in msg.data else SAFE_DISTANCE_REPLY)
        self.get_logger().info(f"Motion stopped for safety ({msg.data}): {reply!r}")
        self._reply_pub.publish(String(data=reply))

    def _in_their_language(self, reply: str) -> str:
        return ARABIC_REPLIES.get(reply, reply) if self._language == ARABIC else reply

    def _on_person_arrived(self, _msg: String) -> None:
        """Someone walked up: greet them, unless the robot is busy answering or already talking with someone."""
        if in_conversation(time.monotonic(), self._last_answer_at):
            self.get_logger().info("Someone came into view during a conversation; not greeting again.")
            return
        if not self._busy.acquire(blocking=False):
            return
        try:
            self.get_logger().info(f"Greeting a person who walked up: {GREETING_FOR_PERSON!r}")
            self._gesture_pub.publish(String(data="wave_right"))
            self._reply_pub.publish(String(data=GREETING_FOR_PERSON))
        finally:
            self._busy.release()

    def _on_speech_ended(self, msg: Float64) -> None:
        self._speech_ended = msg.data

    def _on_transcript(self, msg: String) -> None:
        text = msg.data.strip()
        speech_ended, self._speech_ended = self._speech_ended, None  # typed requests have no spoken end
        if not text:
            return
        self._language = language_of(text)
        if is_stop_request(text):  # at once, even while answering: the reply can follow, the robot must not walk on
            self._motion_pub.publish(String(data="stop"))
            self.get_logger().info("Stop requested; stopping before answering.")
        with self._pending_lock:
            if not self._busy.acquire(blocking=False):
                earlier = self._pending[0] + " " if self._pending else ""
                self._pending = (earlier + text, speech_ended)
                self.get_logger().info("Still answering the previous request; will answer this one next.")
                return
        threading.Thread(target=self._respond, args=(text, speech_ended), daemon=True).start()

    def _log_timing(self, speech_ended: float | None) -> None:
        if self._conversation is None or self._conversation.last_timing is None:
            return
        line = f"Timing: {self._conversation.last_timing.summary()} ({self._backend.last_used or 'commands'})"
        if (since_speech := seconds_since(speech_ended)) is not None:
            line += f"; reply ready {since_speech:.2f} s after you stopped speaking"
        self.get_logger().info(line)

    def _report_usage(self) -> None:
        """Tokens used by this answer and what is left of Groq's limits: logged, and on /voice/usage (numbers only,
        never what was said)."""
        usage, timing = self._backend.last_usage, self._conversation.last_timing
        if usage is not None:
            self.get_logger().info(f"Tokens: {usage.summary()}")
        report = {"backend": self._backend.last_used or "commands", **(asdict(usage) if usage else {}),
                  "steps_s": dict(timing.steps) if timing is not None else {}}
        self._usage_pub.publish(String(data=json.dumps(report)))

    def _respond(self, text: str, speech_ended: float | None = None) -> None:
        try:
            if self._conversation is None:
                return
            if (view := self._surroundings()) is not None:
                self.get_logger().info(f"Camera view for this answer: {view}")
            self._backend.last_used, self._backend.last_usage = "", None
            try:
                response = self._conversation.respond(text)
            except Exception as error:  # network or server failure must not kill the node
                self.get_logger().error(f"No language model could answer: {error}")
                self._log_timing(speech_ended)
                self._reply_pub.publish(String(data=self._in_their_language(NO_ANSWER_REPLY)))
                return
            if self._backend.last_error and self._backend.has_fallback and "waited" not in self._backend.last_error:
                self.get_logger().warn(f"Groq unavailable ({self._backend.last_error}); answered with the local model.")
            elif self._backend.last_error:
                self.get_logger().warn(f"Groq {self._backend.last_error}.")
            self.get_logger().info(
                f"[{self._backend.last_used or 'commands'}] Reply: {response.reply!r} "
                f"gesture={response.gesture} motion={response.motion}"
            )
            self._log_timing(speech_ended)
            self._report_usage()
            self._last_answer_at = time.monotonic()
            self._gesture_pub.publish(String(data=response.gesture))
            if response.motion != "none":
                self._motion_pub.publish(String(data=response.motion))
            self._reply_pub.publish(String(data=response.reply))
        finally:
            with self._pending_lock:
                pending, self._pending = self._pending, None
                if pending is None:
                    self._busy.release()
            if pending is not None:  # keep the lock and answer what was said in the meantime
                self._respond(*pending)


def main() -> None:
    run_node(BrainNode)


if __name__ == "__main__":
    main()
