"""The languages the robot speaks (English and Arabic) and how to tell which one a sentence is in.

Whisper writes Arabic speech in Arabic letters, so every node can tell the language from the text itself: the
brain answers in it, and the voice picks the NVIDIA voice for it, sentence by sentence.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

ENGLISH = "en"
ARABIC = "ar"
# Arabic, Arabic Supplement, Arabic Extended-A and the presentation forms
_ARABIC_LETTER = re.compile("[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]")


def language_of(text: str) -> str:
    """ARABIC when at least half of the words are Arabic ("ما هي شركة Aideology؟"), otherwise ENGLISH."""
    words = re.findall(r"[^\W\d_]+", text)
    arabic = sum(1 for word in words if _ARABIC_LETTER.match(word))
    return ARABIC if words and 2 * arabic >= len(words) else ENGLISH


def has_arabic(text: str) -> bool:
    return bool(_ARABIC_LETTER.search(text))


def choose_language(probabilities: Sequence[tuple[str, float]], allowed: Sequence[str]) -> str:
    """The most likely of the ``allowed`` languages in Whisper's (language, probability) list; the first allowed
    one when Whisper gave none of them."""
    scores = dict(probabilities)
    return max(allowed, key=lambda language: scores.get(language, 0.0)) if any(
        language in scores for language in allowed) else allowed[0]
