"""ROS 2 node: turns transcripts into a spoken reply, a gesture and (when walking) a motion.

Language model: Groq (qwen/qwen3.8-27b by default, key from the GROQ_API_KEY environment variable), falling
back to a local Ollama model. Qwen3-Embedding finds relevant knowledge and memories for every question.
"""

from __future__ import annotations

import os
import threading

from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String

from robot_voice_assistant.assistant_factory import BrainSettings, create_backend, create_conversation
from robot_voice_assistant.conversation import Conversation
from robot_voice_assistant.node_runner import run_node
from robot_voice_assistant.ollama_client import OllamaClient

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

        self._reply_pub = self.create_publisher(String, "/voice/reply", 10)
        self._gesture_pub = self.create_publisher(String, "/robot/gesture", 10)
        self._motion_pub = self.create_publisher(String, "/robot/motion", 10)
        self.create_subscription(String, "/voice/transcript", self._on_transcript, 10)
        self.create_subscription(String, "/voice/reset", self._on_reset, 10)
        self.create_subscription(String, "/voice/forget", self._on_forget, 10)
        threading.Thread(target=self._start, daemon=True).start()

    def _start(self) -> None:
        """Load the embedding model and the knowledge base (takes a little while), then start answering."""
        with self._busy:
            try:
                self._conversation, notes = create_conversation(self._settings, self._backend)
            except Exception as error:  # e.g. no GPU memory for the embedding model
                self.get_logger().error(f"Could not load embeddings ({error}); continuing without them.")
                self._settings.use_embeddings = False
                self._conversation, notes = create_conversation(self._settings, self._backend)
            for note in notes:
                self.get_logger().info(note)
            try:
                OllamaClient(self._settings.ollama_url, self._settings.fallback_model).warm_up()
            except Exception as error:
                self.get_logger().warn(f"Local fallback model not available ({error}). Is Ollama running?")
        walking = "walking enabled" if self._settings.can_walk else "standing in place"
        self.get_logger().info(f"Brain ready ({walking}).")

    def _on_reset(self, _msg: String) -> None:
        with self._busy:
            if self._conversation is not None:
                self._conversation.reset()
        self.get_logger().info("Conversation history cleared.")

    def _on_forget(self, _msg: String) -> None:
        """Delete everything the robot remembers from earlier conversations."""
        with self._busy:
            if self._conversation is not None:
                self._conversation.forget_memories()
        self.get_logger().info("Long-term memory deleted.")

    def _on_transcript(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return
        if not self._busy.acquire(blocking=False):
            self.get_logger().warn("Still busy with the previous request; dropped a new one.")
            return
        threading.Thread(target=self._respond, args=(text,), daemon=True).start()

    def _respond(self, text: str) -> None:
        try:
            if self._conversation is None:
                return
            try:
                response = self._conversation.respond(text)
            except Exception as error:  # network or server failure must not kill the node
                self.get_logger().error(f"No language model could answer: {error}")
                self._reply_pub.publish(String(data="Sorry, I can't think right now. Please try again in a moment."))
                return
            if self._backend.last_error:
                self.get_logger().warn(f"Groq unavailable ({self._backend.last_error}); answered with the local model.")
            self.get_logger().info(
                f"[{self._backend.last_used or 'commands'}] Reply: {response.reply!r} "
                f"gesture={response.gesture} motion={response.motion}"
            )
            self._gesture_pub.publish(String(data=response.gesture))
            if response.motion != "none":
                self._motion_pub.publish(String(data=response.motion))
            self._reply_pub.publish(String(data=response.reply))
        finally:
            self._busy.release()


def main() -> None:
    run_node(BrainNode)


if __name__ == "__main__":
    main()
