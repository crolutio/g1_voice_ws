"""Talk to the robot's brain by typing, without the simulator, microphone or speakers.

Uses exactly the same setup as brain_node: Groq (GROQ_API_KEY) with the local Llama fallback, the knowledge
base, long-term memory and command matching.

    ros2 run robot_voice_assistant chat                        # interactive chat
    ros2 run robot_voice_assistant chat --local                # only the local model, no Groq
    ros2 run robot_voice_assistant chat --questions q.txt      # answer every line of a file
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from robot_voice_assistant.assistant_factory import BrainSettings, create_backend, create_conversation
from robot_voice_assistant.conversation import Conversation
from robot_voice_assistant.llm_backends import FallbackBackend

DEFAULT_KNOWLEDGE_DIR = os.path.expanduser("~/g1_voice_ws/src/robot_voice_assistant/config/knowledge")


def _ask(conversation: Conversation, backend: FallbackBackend, question: str) -> None:
    started = time.monotonic()
    response = conversation.respond(question)
    motion = f", walk: {response.motion}" if response.motion != "none" else ""
    source = backend.last_used or "commands"
    print(f"Robot [{source}, {response.gesture}{motion}, {time.monotonic() - started:.1f}s]: {response.reply}\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", action="store_true", help="use only the local fallback model, not Groq")
    parser.add_argument("--fallback-model", default=BrainSettings.fallback_model)
    parser.add_argument("--knowledge-dir", default=DEFAULT_KNOWLEDGE_DIR)
    parser.add_argument("--no-memory", action="store_true", help="do not read or write the long-term memory")
    parser.add_argument("--no-embeddings", action="store_true", help="no knowledge search, memory or command matching")
    parser.add_argument("--questions", help="file with one question per line; '---' starts a new conversation")
    args = parser.parse_args()

    settings = BrainSettings(
        use_groq=not args.local,
        fallback_model=args.fallback_model,
        knowledge_dir=args.knowledge_dir,
        use_memory=not args.no_memory,
        use_embeddings=not args.no_embeddings,
    )
    backend, description = create_backend(settings)
    print(f"Language model: {description}", flush=True)
    print("Loading...", flush=True)
    conversation, notes = create_conversation(settings, backend)
    for note in notes:
        print(note, flush=True)
    print()

    if args.questions:
        with open(args.questions, encoding="utf-8") as file:
            for line in (line.strip() for line in file):
                if line == "---":
                    conversation.reset()
                    print("--- new conversation ---\n")
                elif line:
                    print(f"You: {line}")
                    _ask(conversation, backend, line)
        return

    print("Type a question and press Enter. '/reset' starts a new conversation, '/forget' deletes the long-term")
    print("memory, Ctrl+D or '/quit' exits.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question in ("/quit", "/exit"):
            return
        if question == "/reset":
            conversation.reset()
            print("(conversation cleared)\n")
        elif question == "/forget":
            conversation.forget_memories()
            print("(long-term memory deleted)\n")
        elif question:
            _ask(conversation, backend, question)


if __name__ == "__main__":
    sys.exit(main())
