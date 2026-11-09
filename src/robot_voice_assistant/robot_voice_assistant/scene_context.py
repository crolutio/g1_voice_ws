"""What the robot sees, in words: YOLO detections turned into a short description, and when to greet a person.

The description goes into the language model's prompt, so "what is this?" can be answered from what the head
camera sees; the greeter decides when someone walking up should hear "Hello, may I help you?".
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

MIN_CONFIDENCE = 0.3
CLOSE_HEIGHT = 0.4  # a box taller than this share of the image is close to the camera
PLURALS = {"person": "people"}
ONE_OF = {"scissors": "a pair of scissors", "glasses": "a pair of glasses"}  # names that are already plural


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    x_center: float  # 0 = left edge of the image, 1 = right edge
    height: float  # share of the image height
    box: tuple[float, float, float, float] | None = None  # (x1, y1, x2, y2), shares of the image width and height


EVERYDAY_NAMES = {"dining table": "table", "cell phone": "phone", "potted plant": "plant", "tv": "television",
                  "sports ball": "ball", "wine glass": "glass", "couch": "sofa"}


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two boxes."""
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


SAME_THING_IOU = 0.7  # boxes this alike are one thing, even with two names (a mug the other model calls a cup)


def merge_detections(per_model: list[list[Detection]], ignore: set[str], same_iou: float = 0.5) -> list[Detection]:
    """Join the detections of several models: everyday names, ``ignore``d labels left out, and the same thing
    seen twice (same name and overlapping boxes, or nearly the same box under another name) counted once,
    keeping the more confident detection."""
    kept: list[Detection] = []
    candidates = [d for detections in per_model for d in detections if d.label not in ignore]
    for detection in sorted(candidates, key=lambda d: d.confidence, reverse=True):
        label = EVERYDAY_NAMES.get(detection.label, detection.label)
        detection = Detection(label, detection.confidence, detection.x_center, detection.height, detection.box)
        duplicate = any(
            other.box is not None and detection.box is not None
            and _overlap(other.box, detection.box) >= (same_iou if other.label == label else SAME_THING_IOU)
            for other in kept
        )
        if not duplicate:
            kept.append(detection)
    return kept


def _where(detection: Detection) -> str:
    if detection.x_center < 0.33:
        side = "on your left"
    elif detection.x_center > 0.67:
        side = "on your right"
    else:
        side = "right in front of you"
    return f"{side}, close" if detection.height > CLOSE_HEIGHT else side


def _article(label: str) -> str:
    return "an" if label[0] in "aeiou" else "a"


def describe_scene(detections: list[Detection]) -> str:
    """One line for the prompt, e.g. "an apple right in front of you, close; a person on your left"."""
    sure = [d for d in detections if d.confidence >= MIN_CONFIDENCE]
    if not sure:
        return "Nothing in particular."
    counts = Counter(d.label for d in sure)
    parts = []
    for label, count in counts.most_common():
        if count == 1:
            only = next(d for d in sure if d.label == label)
            parts.append(f"{ONE_OF.get(label, f'{_article(label)} {label}')} {_where(only)}")
        else:
            parts.append(f"{count} {PLURALS.get(label, label + 's')}")
    return "; ".join(parts)


# Questions answered from what the camera sees right now ("what is this?", "what am I holding?"): their answers
# go out of date as soon as something is moved, so they are not kept in long-term memory.
_ABOUT_THE_VIEW = re.compile(
    r"\b(what|who)('s|\s+is|\s+are)?\s+(this|that|these|those|there|here|around)\b"
    r"|\bwhat(\s+\w+){1,3}\s+(is|are)\s+(this|that|these|those|it)\b|\b(is|are)\s+(this|that|these|those)\b"
    r"|\b(see|seeing|look at|looking at|holding|showing)\b|\bin front of you\b|\bwhat am i\b",
    re.IGNORECASE,
)


def is_about_the_view(question: str) -> bool:
    return bool(_ABOUT_THE_VIEW.search(question))


NEARBY_HEIGHT = 0.33  # a person this tall in the image has walked up to the robot; smaller ones walk past behind


def people_nearby(detections: list[Detection]) -> int:
    """How many people stand close enough to be talking to the robot (not the crowd in the background)."""
    return sum(1 for d in detections if d.label == "person" and d.height >= NEARBY_HEIGHT)


class PersonGreeter:
    """Greets each person who walks up, also when others are already there (a busy expo hall): once more people are
    nearby than were greeted, for ``confirm_frames`` updates in a row, and at most once every ``cooldown_s``
    seconds. Someone counts as gone once fewer people were nearby for ``absent_s`` seconds, so a person hidden
    for a moment is not greeted again."""

    def __init__(self, confirm_frames: int = 2, absent_s: float = 5.0, cooldown_s: float = 10.0) -> None:
        self.confirm_frames, self.absent_s, self.cooldown_s = confirm_frames, absent_s, cooldown_s
        self._greeted = 0  # people nearby when the robot last greeted (and still there)
        self._fewest = 0  # fewest people nearby while there were fewer than greeted
        self._last_all_there: float | None = None  # last time everyone greeted was still nearby
        self._seen_frames = 0
        self._last_greeting: float | None = None

    def update(self, now: float, people_nearby: int) -> bool:
        """True when the robot should greet now."""
        if people_nearby < self._greeted:
            self._fewest = min(self._fewest, people_nearby)
            self._seen_frames = 0
            return False
        if self._last_all_there is not None and self._fewest < self._greeted \
                and now - self._last_all_there >= self.absent_s:
            self._greeted = self._fewest  # the ones missing that long have left
        self._fewest, self._last_all_there = people_nearby, now
        if people_nearby <= self._greeted:
            self._seen_frames = 0
            return False
        self._seen_frames += 1
        if self._seen_frames < self.confirm_frames:
            return False
        if self._last_greeting is not None and now - self._last_greeting < self.cooldown_s:
            return False
        self._greeted, self._last_greeting, self._seen_frames = people_nearby, now, 0
        return True
