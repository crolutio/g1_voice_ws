"""Meaning-based search with text embeddings: knowledge base, long-term memory and command matching."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

KNOWLEDGE_SUFFIXES = (".md", ".txt")


class TextEmbedder(Protocol):
    """Turns texts into unit-length vectors; similar meanings give a high dot product."""

    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


class Qwen3Embedder:
    """Qwen3-Embedding through sentence-transformers (queries use the model's retrieval instruction)."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "cuda") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name, device=device, model_kwargs={"torch_dtype": "auto"})

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, prompt_name="query", normalize_embeddings=True, convert_to_numpy=True)
        return vectors.astype(np.float32)


# --- knowledge base -------------------------------------------------------------------------------


def split_into_passages(text: str, max_chars: int = 600) -> list[str]:
    """Group paragraphs into passages of at most ``max_chars``; very long paragraphs are cut at word boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        while len(paragraph) > max_chars:
            cut = paragraph.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        if paragraph:
            pieces.append(paragraph)
    passages: list[str] = []
    for piece in pieces:
        if passages and len(passages[-1]) + 2 + len(piece) <= max_chars:
            passages[-1] = f"{passages[-1]}\n\n{piece}"
        else:
            passages.append(piece)
    return passages


def _is_title(line: str) -> bool:
    return line.startswith("#") or (line.endswith(":") and len(line) < 80)


def passages_from_document(text: str, default_title: str) -> list[tuple[str, str]]:
    """(title, passage) pairs. Headings and short lines ending in ':' become the title of what follows,
    so a passage like "Its CEO is ..." is embedded together with what "it" refers to."""
    title = default_title
    result: list[tuple[str, str]] = []
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        lines = [line.strip().lstrip("-*").strip() for line in paragraph.splitlines() if line.strip()]
        # Short lines (bullet lists, one fact per line) are separate passages so answers stay precise;
        # running text is cut into passages of a few sentences.
        units = lines if all(len(line) < 300 for line in lines) else split_into_passages(paragraph)
        for unit in units:
            if _is_title(unit):
                title = unit.lstrip("#").strip().rstrip(":").strip()
            elif len(unit) > 3:
                result.append((title, unit))
    return result


@dataclass(frozen=True)
class Passage:
    text: str
    source: str
    score: float = 0.0


class KnowledgeBase:
    def __init__(self, passages: Sequence[Passage], vectors: np.ndarray, embedder: TextEmbedder) -> None:
        self._passages = list(passages)
        self._vectors = vectors
        self._embedder = embedder

    @classmethod
    def load(cls, folder: str | os.PathLike[str], embedder: TextEmbedder, cache_dir: str | os.PathLike[str]) -> KnowledgeBase:
        """Read every .md/.txt file in ``folder``; embeddings are cached until a file or the model changes."""
        folder = Path(folder)
        passages: list[Passage] = []
        embedded_texts: list[str] = []
        for path in sorted(folder.glob("*")) if folder.is_dir() else []:
            if path.suffix.lower() in KNOWLEDGE_SUFFIXES and path.is_file():
                default_title = path.stem.replace("_", " ").replace("-", " ")
                for title, text in passages_from_document(path.read_text(encoding="utf-8"), default_title):
                    passages.append(Passage(text, path.name))
                    embedded_texts.append(f"{title}: {text}")
        if not passages:
            return cls([], np.zeros((0, 1), dtype=np.float32), embedder)

        fingerprint = hashlib.sha256(
            json.dumps([embedder.name, [(p.source, t) for p, t in zip(passages, embedded_texts)]]).encode()
        ).hexdigest()[:20]
        cache_path = Path(cache_dir) / f"knowledge_{fingerprint}.npy"
        if cache_path.exists():
            vectors = np.load(cache_path)
        else:
            vectors = embedder.embed_documents(embedded_texts)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, vectors)
        return cls(passages, vectors, embedder)

    def __len__(self) -> int:
        return len(self._passages)

    def search(self, question: str, top_k: int = 4, min_score: float = 0.35) -> list[Passage]:
        if not self._passages:
            return []
        scores = self._vectors @ self._embedder.embed_queries([question])[0]
        best = np.argsort(-scores)[:top_k]
        return [
            Passage(self._passages[i].text, self._passages[i].source, float(scores[i]))
            for i in best
            if scores[i] >= min_score
        ]


# --- long-term memory -----------------------------------------------------------------------------

_SMALL_TALK = re.compile(r"^(thanks?|thank you|ok(ay)?|yes|no|hi|hello|hey|bye|goodbye|stop)\b[\s!.,]*$", re.IGNORECASE)


@dataclass(frozen=True)
class MemoryEntry:
    time: float
    user: str
    reply: str
    score: float = 0.0


class LongTermMemory:
    """Past exchanges stored on disk (one JSON line each), recalled by meaning."""

    def __init__(self, path: str | os.PathLike[str], embedder: TextEmbedder, max_entries: int = 2000) -> None:
        self._path = Path(path)
        self._embedder = embedder
        self._max_entries = max_entries
        self._entries: list[MemoryEntry] = []
        vectors: list[list[float]] = []
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record.get("model") == embedder.name:
                    self._entries.append(MemoryEntry(record["time"], record["user"], record["reply"]))
                    vectors.append(record["vector"])
        self._vectors = np.array(vectors, dtype=np.float32) if vectors else np.zeros((0, 1), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, user: str, reply: str) -> None:
        """Remember an exchange (greetings, thanks and other small talk are skipped)."""
        if _SMALL_TALK.match(user.strip()) or len(user.split()) < 3:
            return
        vector = self._embedder.embed_documents([f"{user}\n{reply}"])[0]
        self._entries.append(MemoryEntry(time.time(), user, reply))
        self._vectors = vector[None, :] if len(self._entries) == 1 else np.vstack([self._vectors, vector])
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]
            self._vectors = self._vectors[-self._max_entries :]
            self._rewrite()
        else:
            self._append(self._entries[-1], vector)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.45) -> list[MemoryEntry]:
        if not self._entries:
            return []
        scores = self._vectors @ self._embedder.embed_queries([query])[0]
        best = np.argsort(-scores)[:top_k]
        return [
            MemoryEntry(self._entries[i].time, self._entries[i].user, self._entries[i].reply, float(scores[i]))
            for i in best
            if scores[i] >= min_score
        ]

    def clear(self) -> None:
        self._entries = []
        self._vectors = np.zeros((0, 1), dtype=np.float32)
        self._path.unlink(missing_ok=True)

    def _record(self, entry: MemoryEntry, vector: np.ndarray) -> str:
        return json.dumps(
            {"time": entry.time, "user": entry.user, "reply": entry.reply, "model": self._embedder.name,
             "vector": [round(float(v), 5) for v in vector]}
        )

    def _append(self, entry: MemoryEntry, vector: np.ndarray) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as file:
            file.write(self._record(entry, vector) + "\n")

    def _rewrite(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines = [self._record(e, v) for e, v in zip(self._entries, self._vectors)]
        self._path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- command matching -----------------------------------------------------------------------------


class CommandMatcher:
    """Finds the command whose example phrases mean the same as what the person said."""

    def __init__(self, phrases: Mapping[str, Sequence[str]], embedder: TextEmbedder, threshold: float = 0.7) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self._commands = [command for command, examples in phrases.items() for _ in examples]
        self._vectors = embedder.embed_documents([text for examples in phrases.values() for text in examples])
        self._embedder = embedder
        self._threshold = threshold

    def match(self, text: str, allowed: Sequence[str] | None = None) -> str | None:
        scores = self._vectors @ self._embedder.embed_queries([text])[0]
        for index in np.argsort(-scores):
            if scores[index] < self._threshold:
                return None
            if allowed is None or self._commands[index] in allowed:
                return self._commands[index]
        return None


# --- context for the conversation ---------------------------------------------------------------


class SemanticContext:
    """Per question: the most relevant knowledge passages and memories, formatted for the system prompt."""

    def __init__(
        self,
        knowledge: KnowledgeBase | None,
        memory: LongTermMemory | None,
        recent_window: int = 10,
        knowledge_min_score: float = 0.35,
        memory_min_score: float = 0.45,
    ) -> None:
        self._knowledge = knowledge
        self._memory = memory
        self._recent: list[str] = []  # this session's last exchanges are already in the chat history
        self._recent_window = recent_window
        self._knowledge_min_score = knowledge_min_score
        self._memory_min_score = memory_min_score

    def context_for(self, question: str) -> str:
        sections = []
        if self._knowledge is not None:
            passages = self._knowledge.search(question, top_k=4, min_score=self._knowledge_min_score)
            if passages:
                sections.append("\n".join(f"- {p.text} (from {p.source})" for p in passages))
        if self._memory is not None:
            memories = [
                m for m in self._memory.search(question, top_k=5, min_score=self._memory_min_score)
                if m.user not in self._recent
            ][:3]
            if memories:
                lines = [
                    f"- On {time.strftime('%d %B %Y', time.localtime(m.time))} the person said \"{m.user}\" "
                    f"and you answered \"{m.reply}\""
                    for m in memories
                ]
                sections.append("Things you remember from earlier conversations with this person:\n" + "\n".join(lines))
        return "\n\n".join(sections)

    def remember(self, user: str, reply: str) -> None:
        if self._memory is not None:
            self._memory.add(user, reply)
        self._recent = (self._recent + [user])[-self._recent_window :]

    def forget(self) -> None:
        if self._memory is not None:
            self._memory.clear()
        self._recent = []
