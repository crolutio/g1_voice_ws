"""Decides which transcripts are addressed to the robot, so nearby conversations are ignored.

Saying a wake word ("robot") starts a conversation. From then on every sentence is answered, without the
wake word, until the person ends it ("goodbye", "stop listening", ...) or, if ``idle_timeout_s`` is set,
nobody has spoken for that long.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

GREETING_FOR_BARE_WAKE_WORD = "Hello!"
# What Whisper writes for "robot" in some voices; counts only at the start of a sentence
DEFAULT_SOUND_ALIKES = ("robert",)

# Short sentences that end the conversation. "Stop" alone is not one of them: it stops walking.
END_PHRASES = (
    "goodbye",
    "good bye",
    "bye bye",
    "bye robot",
    "bye for now",
    "see you later",
    "see you",
    "stop listening",
    "stop the conversation",
    "end the conversation",
    "end conversation",
    "that's all",
    "that is all",
    "that will be all",
    "go to sleep",
    "you can sleep",
)
MAX_END_SENTENCE_WORDS = 6
_NON_WORD = re.compile(r"[^a-z0-9' ]+")


def _normalize(text: str) -> str:
    return " ".join(_NON_WORD.sub(" ", text.lower()).split())


# Words people (or Whisper) put around the wake word that are not part of the request
_BEFORE_WAKE_WORD = frozenset({"", "a", "oh", "hey", "hi", "hello", "ok", "okay", "the"})
_GREETINGS = frozenset({"", "hey", "hi", "hello", "hey there", "hi there", "hello there", "ok", "okay"})


def _wake_word_pattern(wake_word: str) -> re.Pattern[str]:
    words = [re.escape(word) for word in wake_word.lower().split()]
    # Whole words only, allowing spaces or hyphens between parts ("g one", "g-one"), and Whisper's "robots".
    plural = "s?" if len(words) == 1 and len(wake_word) >= 4 else ""
    return re.compile(r"\b" + r"[\s\-]*".join(words) + plural + r"\b", re.IGNORECASE)


def ends_conversation(text: str) -> bool:
    normalized = _normalize(text)
    if len(normalized.split()) > MAX_END_SENTENCE_WORDS:
        return False
    return any(re.search(rf"\b{re.escape(phrase)}\b", normalized) for phrase in END_PHRASES)


class WakeWordGate:
    def __init__(
        self,
        wake_words: Sequence[str],
        idle_timeout_s: float = 0.0,
        wake_word_wait_s: float = 3.0,
        sound_alikes: Sequence[str] = (),
    ) -> None:
        """``idle_timeout_s`` 0 = a conversation lasts until the person ends it.

        ``sound_alikes``: what speech recognition sometimes writes for the wake word ("Robert" for "robot").
        They only count at the start of a sentence, where people address the robot, not in the middle
        ("I told Robert ...").

        ``wake_word_wait_s``: after the wake word alone ("Robot..."), how long to wait for the question that
        usually follows before greeting the person.
        """
        self._patterns = [_wake_word_pattern(w) for w in wake_words if w.strip()]
        fillers = "|".join(sorted(word for word in _BEFORE_WAKE_WORD if word))
        self._sound_alike_patterns = [
            re.compile(rf"^\s*(?:(?:{fillers})[\s,]+)?" + _wake_word_pattern(w).pattern, re.IGNORECASE)
            for w in sound_alikes if w.strip()
        ]
        self._idle_timeout_s = idle_timeout_s
        self._wake_word_wait_s = wake_word_wait_s
        self.in_conversation = False
        self._last_activity_s = float("-inf")
        self._greet_at_s: float | None = None

    def note_reply_finished(self, now_s: float) -> None:
        """The idle time of a conversation counts from the robot's last reply."""
        self._last_activity_s = now_s

    def poll(self, now_s: float, speech_in_progress: bool = False) -> str | None:
        """The greeting to answer a bare wake word with, once no question followed it in time.

        ``speech_in_progress``: the person is speaking right now (probably the question), so keep waiting.
        """
        if self._greet_at_s is not None and speech_in_progress:
            self._greet_at_s = max(self._greet_at_s, now_s + 1.0)
        if self._greet_at_s is not None and now_s >= self._greet_at_s:
            self._greet_at_s = None
            return GREETING_FOR_BARE_WAKE_WORD
        return None

    def accept(self, transcript: str, now_s: float) -> str | None:
        """Return the request text if it is addressed to the robot, otherwise None."""
        text = transcript.strip()
        if not self._patterns:
            return text or None
        if self.in_conversation and self._idle_timeout_s > 0 and now_s - self._last_activity_s > self._idle_timeout_s:
            self.in_conversation = False
        request = self._without_wake_word(text)
        if request is None and not self.in_conversation:
            return None
        self.in_conversation = True
        self._last_activity_s = now_s
        if request == "":  # only the wake word: the question usually follows after a short pause
            self._greet_at_s = now_s + self._wake_word_wait_s
            return None
        self._greet_at_s = None
        request = request if request is not None else text
        if ends_conversation(request) or ends_conversation(text):
            self.in_conversation = False
        return request or None

    def _without_wake_word(self, text: str) -> str | None:
        """The text without the wake word ("" if nothing else was said), or None if there is no wake word."""
        for pattern in [*self._patterns, *self._sound_alike_patterns]:
            match = pattern.search(text)
            if match is None:
                continue
            before = text[: match.start()].strip(" ,.!?-")
            after = text[match.end() :].lstrip(" ,.!?-").rstrip()
            if _normalize(before) in _BEFORE_WAKE_WORD:
                before = ""
            if _normalize(after) in _GREETINGS:
                after = ""
            return " ".join(part for part in (before, after) if part)
        return None
