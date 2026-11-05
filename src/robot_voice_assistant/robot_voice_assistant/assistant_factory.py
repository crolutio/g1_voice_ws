"""Builds the robot's brain (language models, embeddings, knowledge, memory) from settings.

Shared by brain_node (ROS 2) and the typed chat tool, so both behave the same.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from robot_voice_assistant.conversation import Conversation
from robot_voice_assistant.gestures import GESTURE_DESCRIPTIONS, GESTURE_PHRASES
from robot_voice_assistant.llm_backends import DEFAULT_GROQ_MODEL, FallbackBackend, GroqClient
from robot_voice_assistant.motions import MOTION_DESCRIPTIONS, MOTION_PHRASES
from robot_voice_assistant.ollama_client import OllamaClient

DEFAULT_DATA_DIR = os.path.expanduser("~/.local/share/robot_voice_assistant")


@dataclass
class BrainSettings:
    robot_name: str = "R1"
    can_walk: bool = True
    history_turns: int = 10
    # Language models: Groq in the cloud, a local Ollama model when Groq fails.
    use_groq: bool = True
    groq_model: str = DEFAULT_GROQ_MODEL
    fallback_model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434"
    # Embeddings: knowledge base search, long-term memory and command matching.
    use_embeddings: bool = True
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_device: str = "cuda"
    knowledge_dir: str = ""
    knowledge_min_score: float = 0.40
    use_memory: bool = True
    memory_file: str = field(default_factory=lambda: os.path.join(DEFAULT_DATA_DIR, "memory.jsonl"))
    memory_min_score: float = 0.40
    command_match_threshold: float = 0.68
    cache_dir: str = field(default_factory=lambda: os.path.join(DEFAULT_DATA_DIR, "cache"))


def create_backend(settings: BrainSettings, environment: Mapping[str, str] = os.environ) -> tuple[FallbackBackend, str]:
    """The chat backend and a one-line description of it for the log."""
    fallback = OllamaClient(settings.ollama_url, settings.fallback_model)
    api_key = environment.get("GROQ_API_KEY", "").strip()
    if settings.use_groq and api_key:
        primary = GroqClient(api_key, settings.groq_model)
        return FallbackBackend(primary, fallback), (
            f"Groq {settings.groq_model}, falling back to local {settings.fallback_model}"
        )
    if settings.use_groq:
        reason = "GROQ_API_KEY is not set, so only the local model is used"
        return FallbackBackend(None, fallback), f"local {settings.fallback_model} ({reason})"
    return FallbackBackend(None, fallback), f"local {settings.fallback_model}"


def create_conversation(
    settings: BrainSettings, backend: FallbackBackend, surroundings: Callable[[], str | None] | None = None
) -> tuple[Conversation, list[str]]:
    """The conversation, plus notes for the log about what was loaded (knowledge passages, memories)."""
    from robot_voice_assistant.semantic import CommandMatcher, KnowledgeBase, LongTermMemory, Qwen3Embedder, SemanticContext

    notes: list[str] = []
    retriever = None
    matcher = None
    if settings.use_embeddings:
        embedder = Qwen3Embedder(settings.embedding_model, settings.embedding_device)
        knowledge = None
        if settings.knowledge_dir:
            knowledge = KnowledgeBase.load(settings.knowledge_dir, embedder, settings.cache_dir)
            notes.append(f"knowledge base: {len(knowledge)} passages from {settings.knowledge_dir}")
        memory = None
        if settings.use_memory:
            memory = LongTermMemory(settings.memory_file, embedder)
            notes.append(f"long-term memory: {len(memory)} exchanges in {settings.memory_file}")
        retriever = SemanticContext(
            knowledge,
            memory,
            recent_window=settings.history_turns,
            knowledge_min_score=settings.knowledge_min_score,
            memory_min_score=settings.memory_min_score,
        )
        phrases = {**GESTURE_PHRASES, **(MOTION_PHRASES if settings.can_walk else {})}
        matcher = CommandMatcher(phrases, embedder, threshold=settings.command_match_threshold)
        notes.append(f"command matching: {sum(map(len, phrases.values()))} example phrases")
    conversation = Conversation(
        backend,
        robot_name=settings.robot_name,
        gesture_descriptions=GESTURE_DESCRIPTIONS,
        history_turns=settings.history_turns,
        motion_descriptions=MOTION_DESCRIPTIONS if settings.can_walk else None,
        retriever=retriever,
        command_matcher=matcher,
        surroundings=surroundings,
    )
    return conversation, notes
