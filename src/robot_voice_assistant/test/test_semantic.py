import re
import zlib
from pathlib import Path

import numpy as np
import pytest

from robot_voice_assistant.semantic import CommandMatcher, KnowledgeBase, LongTermMemory, split_into_passages

VOCABULARY_SIZE = 512


class WordOverlapEmbedder:
    """Stand-in for Qwen3-Embedding: texts sharing words get similar vectors."""

    name = "word-overlap-test"

    def __init__(self) -> None:
        self.document_calls = 0

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), VOCABULARY_SIZE), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                if len(word) > 2:
                    vectors[row, zlib.crc32(word.encode()) % VOCABULARY_SIZE] += 1.0  # hash() differs per run
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.document_calls += 1
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)


# --- passages -----------------------------------------------------------------


def test_split_into_passages_keeps_short_paragraphs_together_and_splits_long_ones() -> None:
    text = "# Title\n\nFirst paragraph.\n\nSecond paragraph.\n\n" + "word " * 300
    passages = split_into_passages(text, max_chars=400)
    assert passages[0].startswith("# Title")
    assert "Second paragraph." in passages[0]
    assert all(len(p) <= 400 for p in passages)
    assert len(passages) >= 3


# --- knowledge base ---------------------------------------------------------------


def _write_knowledge(folder: Path) -> None:
    (folder / "aideology.md").write_text(
        "Aideology is an AI infrastructure company based in Dubai.\n\n"
        "The CEO of Aideology is Jorge Colotto.\n",
        encoding="utf-8",
    )
    (folder / "office.txt").write_text("The office cafeteria serves lunch from noon until two.\n", encoding="utf-8")


def test_knowledge_base_finds_the_relevant_passage(tmp_path: Path) -> None:
    _write_knowledge(tmp_path)
    knowledge = KnowledgeBase.load(tmp_path, WordOverlapEmbedder(), cache_dir=tmp_path / "cache")
    results = knowledge.search("Who is the CEO of Aideology?", top_k=1)
    assert results[0].text == "The CEO of Aideology is Jorge Colotto."
    assert results[0].source == "aideology.md"


def test_knowledge_base_returns_nothing_for_unrelated_questions(tmp_path: Path) -> None:
    _write_knowledge(tmp_path)
    knowledge = KnowledgeBase.load(tmp_path, WordOverlapEmbedder(), cache_dir=tmp_path / "cache")
    assert knowledge.search("Why is the sky blue?", top_k=3, min_score=0.3) == []


def test_knowledge_index_is_cached_until_files_change(tmp_path: Path) -> None:
    _write_knowledge(tmp_path)
    embedder = WordOverlapEmbedder()
    KnowledgeBase.load(tmp_path, embedder, cache_dir=tmp_path / "cache")
    KnowledgeBase.load(tmp_path, embedder, cache_dir=tmp_path / "cache")
    assert embedder.document_calls == 1
    (tmp_path / "office.txt").write_text("Lunch moved to one o'clock.\n", encoding="utf-8")
    KnowledgeBase.load(tmp_path, embedder, cache_dir=tmp_path / "cache")
    assert embedder.document_calls == 2


def test_empty_knowledge_folder_is_allowed(tmp_path: Path) -> None:
    knowledge = KnowledgeBase.load(tmp_path, WordOverlapEmbedder(), cache_dir=tmp_path / "cache")
    assert knowledge.search("anything") == []


# --- long-term memory -------------------------------------------------------------


def test_memory_recalls_a_relevant_earlier_exchange(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path / "memory.jsonl", WordOverlapEmbedder())
    memory.add("My favourite colour is green.", "Green is a lovely colour!")
    memory.add("What is the capital of France?", "The capital of France is Paris.")
    recalled = memory.search("Do you remember my favourite colour?", top_k=1)
    assert recalled[0].user == "My favourite colour is green."


def test_memory_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "memory.jsonl"
    LongTermMemory(path, WordOverlapEmbedder()).add("My dog is called Bruno.", "Bruno is a great name!")
    reloaded = LongTermMemory(path, WordOverlapEmbedder())
    assert reloaded.search("What is my dog called?", top_k=1, min_score=0.3)[0].user == "My dog is called Bruno."


def test_memory_skips_small_talk_and_can_be_cleared(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path / "memory.jsonl", WordOverlapEmbedder())
    memory.add("Thanks!", "You're welcome!")
    assert len(memory) == 0
    memory.add("My name is Radhika.", "Nice to meet you, Radhika!")
    memory.clear()
    assert len(memory) == 0
    assert LongTermMemory(tmp_path / "memory.jsonl", WordOverlapEmbedder()).search("my name") == []


def test_memory_keeps_only_the_newest_entries(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path / "memory.jsonl", WordOverlapEmbedder(), max_entries=3)
    for i in range(5):
        memory.add(f"Remember the number {i} please", f"Okay, number {i}.")
    assert len(memory) == 3
    assert all("number 0" not in entry.user for entry in memory.search("number", top_k=5, min_score=0.0))


# --- command matching --------------------------------------------------------------


COMMAND_PHRASES = {
    "walk_forward": ["come here", "walk towards me", "come over to me"],
    "turn_left": ["turn left", "turn to your left"],
    "wave_right": ["wave at me", "say hello with your hand"],
}


def test_command_matcher_recognises_a_reworded_command() -> None:
    matcher = CommandMatcher(COMMAND_PHRASES, WordOverlapEmbedder(), threshold=0.5)
    assert matcher.match("could you come over here to me") == "walk_forward"
    assert matcher.match("please turn towards your left") == "turn_left"


def test_command_matcher_ignores_ordinary_questions() -> None:
    matcher = CommandMatcher(COMMAND_PHRASES, WordOverlapEmbedder(), threshold=0.5)
    assert matcher.match("What is the capital of India?") is None


def test_command_matcher_can_be_limited_to_some_commands() -> None:
    matcher = CommandMatcher(COMMAND_PHRASES, WordOverlapEmbedder(), threshold=0.5)
    assert matcher.match("wave at me", allowed=("walk_forward", "turn_left")) is None


@pytest.mark.parametrize("threshold", [0.0, 1.5])
def test_command_matcher_rejects_bad_thresholds(threshold: float) -> None:
    with pytest.raises(ValueError):
        CommandMatcher(COMMAND_PHRASES, WordOverlapEmbedder(), threshold=threshold)


def test_semantic_context_combines_knowledge_and_older_memories(tmp_path: Path) -> None:
    from robot_voice_assistant.semantic import SemanticContext

    _write_knowledge(tmp_path)
    embedder = WordOverlapEmbedder()
    memory = LongTermMemory(tmp_path / "memory.jsonl", embedder)
    memory.add("Who is the CEO of Aideology again?", "The CEO is Jorge Colotto.")
    context = SemanticContext(KnowledgeBase.load(tmp_path, embedder, cache_dir=tmp_path / "cache"), memory)
    text = context.context_for("Who is the CEO of Aideology?")
    assert "The CEO of Aideology is Jorge Colotto. (from aideology.md)" in text
    assert "Things you remember" in text
    # An exchange from this session is already in the chat history, so it is not repeated as a memory.
    context.remember("Who is the CEO of Aideology again?", "The CEO is Jorge Colotto.")
    assert "Things you remember" not in context.context_for("Who is the CEO of Aideology?")


def test_command_phrases_refer_to_real_commands() -> None:
    from robot_voice_assistant.gestures import GESTURE_DESCRIPTIONS, GESTURE_PHRASES
    from robot_voice_assistant.motions import MOTION_DESCRIPTIONS, MOTION_PHRASES

    assert set(GESTURE_PHRASES) <= set(GESTURE_DESCRIPTIONS)
    assert set(MOTION_PHRASES) <= set(MOTION_DESCRIPTIONS)


class RecordingEmbedder(WordOverlapEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.documents: list[str] = []

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.documents.extend(texts)
        return super().embed_documents(texts)


def test_passages_are_embedded_with_their_document_title(tmp_path: Path) -> None:
    (tmp_path / "company.md").write_text("About Aideology:\n- Its founder and CEO is Jorge Colotto.\n", encoding="utf-8")
    embedder = RecordingEmbedder()
    knowledge = KnowledgeBase.load(tmp_path, embedder, cache_dir=tmp_path / "cache")
    assert embedder.documents == ["About Aideology: Its founder and CEO is Jorge Colotto."]
    # The prompt shows the passage itself, and the title line is not a passage of its own.
    assert [p.text for p in knowledge.search("Who is the CEO of Aideology?", min_score=0.0)] == [
        "Its founder and CEO is Jorge Colotto."
    ]


def test_markdown_heading_is_used_as_title(tmp_path: Path) -> None:
    (tmp_path / "office.md").write_text("# Dubai office\n\nLunch is served at noon.\n", encoding="utf-8")
    embedder = RecordingEmbedder()
    KnowledgeBase.load(tmp_path, embedder, cache_dir=tmp_path / "cache")
    assert embedder.documents == ["Dubai office: Lunch is served at noon."]


def test_a_long_bullet_list_is_split_into_whole_lines() -> None:
    from robot_voice_assistant.semantic import passages_from_document

    bullets = "\n".join(f"- Fact number {i} is a complete sentence about the company and its work." for i in range(20))
    passages = passages_from_document("About the company:\n" + bullets, "doc")
    assert len(passages) == 20
    assert all(text.startswith("Fact number") and text.endswith(".") for _, text in passages)
