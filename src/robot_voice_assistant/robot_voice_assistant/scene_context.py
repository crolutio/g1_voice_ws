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


NEVER_HELD = {"person", "chair", "table", "television", "sofa", "bed", "plant", "bench", "refrigerator", "car",
              "tie", "backpack", "handbag"}  # the last ones are worn: YOLO finds them on a visitor's clothes


def _name(label: str) -> str:
    return ONE_OF.get(label, f"{_article(label)} {label}")


def _inside(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _nearby_people(detections: list[Detection]) -> list[tuple[int, Detection]]:
    return [(i, d) for i, d in enumerate(detections)
            if d.label == "person" and d.box is not None and d.height >= NEARBY_HEIGHT]


def _holder(thing: Detection, people: list[tuple[int, Detection]]) -> int | None:
    """The index of the close person holding ``thing`` (its middle in their box, and small next to them)."""
    if EVERYDAY_NAMES.get(thing.label, thing.label) in NEVER_HELD or thing.box is None or thing.height > 0.5:
        return None
    middle = ((thing.box[0] + thing.box[2]) / 2, (thing.box[1] + thing.box[3]) / 2)
    holders = [(i, p) for i, p in people if _inside(middle, p.box) and thing.height < 0.6 * p.height]
    if not holders:
        return None
    return max(holders, key=lambda holder: holder[1].height)[0]  # the closest of them


def held_objects(detections: list[Detection]) -> dict[int, list[Detection]]:
    """What people close to the robot hold: {index of the person in ``detections``: the objects in their hands}.
    An object is held when its middle lies in the box of a person who stands close (``NEARBY_HEIGHT``) and it is
    small next to them; furniture and screens behind someone are never held."""
    people = _nearby_people(detections)
    held: dict[int, list[Detection]] = {}
    for thing in detections:
        i = _holder(thing, people)
        if i is not None:
            held.setdefault(i, []).append(thing)
    return held


HELD_KEEP_S = 3.0  # what was seen held in the last this many seconds votes (about 4 frames at 1.5 Hz)


class HeldObjectMemory:
    """Steadies what a close visitor is said to hold: the names seen held in the last ``keep_s`` seconds vote, and
    the most common one is said. A small object in a hand is missed in some frames and misnamed in others (a banana
    called an apple once), and a question is answered from whichever frame is the latest. Forgotten when nobody has
    been close for ``keep_s``; a frame in which the detector misses the close visitor keeps them, and what they hold.
    One held thing per view: a visitor shows one thing at a time."""

    def __init__(self, keep_s: float = HELD_KEEP_S) -> None:
        self.keep_s = keep_s
        self._seen: list[tuple[float, Detection]] = []  # (when, the thing held), newest last
        self._visitor: Detection | None = None  # the closest person last seen close, and when
        self._visitor_at = 0.0

    def update(self, now: float, detections: list[Detection]) -> list[Detection]:
        close = _nearby_people(detections)
        if close:
            self._visitor, self._visitor_at = max(close, key=lambda person: person[1].height)[1], now
        elif self._visitor is not None and now - self._visitor_at <= self.keep_s:
            detections = [*detections, self._visitor]  # missed for a moment: they are still standing there
        else:
            self._seen, self._visitor = [], None  # nobody close for a while: they have gone
            return detections
        held = [thing for things in held_objects(detections).values() for thing in things]
        self._seen = [(at, thing) for at, thing in self._seen if now - at <= self.keep_s]
        self._seen += [(now, thing) for thing in held]
        if not self._seen:
            return detections
        votes = Counter(thing.label for _, thing in self._seen)
        top = max(votes.values())
        # the winner; on a tie, the name seen last. Its latest box stands for it.
        winner = next(thing for _, thing in reversed(self._seen) if votes[thing.label] == top)
        held_ids = {id(thing) for thing in held}
        return [d for d in detections if id(d) not in held_ids] + [winner]


LOOK_CLOSER_MARGIN = 0.15  # of the person's width, on every side
LOOK_CLOSER_DEPTH = 0.65  # of the person's height, from the top: the hands held up to show something, not the legs


CLOSER_LOOKS = 3  # people looked at again at most; a crowd at the stall stands side by side


def closer_looks(detections: list[Detection], limit: int = CLOSER_LOOKS) -> list[tuple[float, float, float, float]]:
    """The parts of the image to look at again, enlarged: the upper body of each person who has walked up (the
    closest first), with a margin; small things in their hands are found far better there than in the whole view.
    Empty when nobody is close."""
    people = sorted(_nearby_people(detections), key=lambda person: person[1].height, reverse=True)[:limit]
    crops = []
    for _, person in people:
        x1, y1, x2, y2 = person.box
        margin = LOOK_CLOSER_MARGIN * (x2 - x1)
        bottom = y1 + LOOK_CLOSER_DEPTH * (y2 - y1) + margin
        crops.append((max(0.0, x1 - margin), max(0.0, y1 - margin), min(1.0, x2 + margin), min(1.0, bottom)))
    return crops


def from_closer_look(detections: list[Detection], crop: tuple[float, float, float, float]) -> list[Detection]:
    """Detections made in ``crop`` (as shares of the crop) placed back in the whole view; people are left out, as
    the whole view already counts them (a cut-off body would be counted twice)."""
    cx, cy, width, height = crop[0], crop[1], crop[2] - crop[0], crop[3] - crop[1]
    placed = []
    for d in detections:
        if d.label == "person" or d.box is None:
            continue
        box = (cx + d.box[0] * width, cy + d.box[1] * height, cx + d.box[2] * width, cy + d.box[3] * height)
        placed.append(Detection(d.label, d.confidence, (box[0] + box[2]) / 2, box[3] - box[1], box))
    return placed


MIN_NAMING_SCORE = 0.15  # the classifier's share for its best word, among all its words
# What YOLO calls small things held up in a hand that are not those (stall demo, 2026-11-10: an apple and a banana
# are a "sports ball" or a "ball", scissors a "tennis racket", a mug a "cell phone"). Its specific names (bottle,
# mug, cup) were right every time, and CLIP, which must pick one of its words, got some of those wrong (a yellow
# mustard bottle a "banana"), so only these are named again.
UNSURE_HELD_NAMES = frozenset({"sports ball", "ball", "tennis racket", "baseball glove", "baseball bat", "frisbee",
                               "kite", "umbrella", "cell phone", "phone"})


def name_held_things(per_model: list[list[Detection]], classify, unsure: frozenset[str] = UNSURE_HELD_NAMES,
                     min_score: float = MIN_NAMING_SCORE) -> list[list[Detection]]:
    """Name again what close people hold when YOLO gave it one of its ``unsure`` names, with ``classify(box) ->
    {word: share}`` (CLIP: it tells a red apple from the "sports ball" YOLO calls it). The best word replaces
    YOLO's name unless the classifier is unsure too; YOLO's confidence that something is there is kept."""
    people = _nearby_people([d for detections in per_model for d in detections])
    named = []
    for detections in per_model:
        renamed = []
        for d in detections:
            if d.label in unsure and _holder(d, people) is not None:
                shares = classify(d.box)
                word = max(shares, key=shares.get) if shares else None
                if word is not None and shares[word] >= min_score:
                    d = Detection(word, d.confidence, d.x_center, d.height, d.box)
            renamed.append(d)
        named.append(renamed)
    return named


def describe_scene(detections: list[Detection]) -> str:
    """One line for the prompt, e.g. "a person right in front of you, close, holding an apple; a chair on your
    left"."""
    sure = [d for d in detections if d.confidence >= MIN_CONFIDENCE]
    if not sure:
        return "Nothing in particular."
    held = held_objects(sure)
    in_hands = {id(thing) for things in held.values() for thing in things}
    parts = []
    for i, things in held.items():
        names = " and ".join(_name(thing.label) for thing in things)
        parts.append(f"a person {_where(sure[i])}, holding {names}")
    rest = [d for i, d in enumerate(sure) if i not in held and id(d) not in in_hands]
    counts = Counter(d.label for d in rest)
    for label, count in counts.most_common():
        if count == 1:
            only = next(d for d in rest if d.label == label)
            parts.append(f"{_name(label)} {_where(only)}")
        else:
            parts.append(f"{count} {PLURALS.get(label, label + 's')}")
    return "; ".join(parts)


# Questions answered from what the camera sees right now ("what is this?", "what am I holding?"): their answers
# go out of date as soon as something is moved, so they are not kept in long-term memory.
_ABOUT_THE_VIEW = re.compile(
    r"\b(what|who)('s|\s+is|\s+are)?\s+(this|that|these|those|there|here|around)\b"
    r"|\bwhat(\s+\w+){1,3}\s+(is|are)\s+(this|that|these|those|it)\b|\b(is|are)\s+(this|that|these|those)\b"
    r"|\b(see|seeing|look at|looking at|holding|showing)\b|\bin front of you\b|\bwhat am i\b"
    r"|\b(in|on) my hands?\b|\bwhat have i got\b",
    re.IGNORECASE,
)


_ABOUT_THEIR_HAND = re.compile(r"\b(in|on) my hands?\b|\bholding\b|\bwhat have i got\b|\bwhat am i\b", re.IGNORECASE)
NOTHING_SEEN_HELD = ("You cannot see anything in their hand. Do not guess what it is: say you cannot make it out "
                     "and ask them to hold it up in front of you.")


def about_their_hand_unseen(question: str, view: str) -> bool:
    """The person asks what they hold, and the view shows nobody close holding anything: the answer must not
    guess (a small language model would name something likely, such as a phone)."""
    return bool(_ABOUT_THEIR_HAND.search(question)) and "holding" not in view


def is_about_the_view(question: str) -> bool:
    return bool(_ABOUT_THE_VIEW.search(question))


NEARBY_HEIGHT = 0.33  # a person this tall in the image has walked up to the robot; smaller ones walk past behind


def people_nearby(detections: list[Detection]) -> int:
    """How many people stand close enough to be talking to the robot (not the crowd in the background)."""
    return sum(1 for d in detections if d.label == "person" and d.height >= NEARBY_HEIGHT)


STALL_SIDES = (0.2, 0.8)  # a person whose middle is between these shares of the width stands in front of the robot
STALL_CONFIRM_FRAMES = 3  # about 2 s at 1.5 Hz: a visitor stops at the stall, a passer-by crosses in a frame or two


def people_at_the_stall(detections: list[Detection]) -> int:
    """How many people stand close and in front of the robot: visitors at its stall. In a busy hall, people walking
    past also come close, at the sides of the view; greeting them would use up the greeting meant for the visitor
    who stops a moment later."""
    return sum(1 for d in detections if d.label == "person" and d.height >= NEARBY_HEIGHT
               and STALL_SIDES[0] <= d.x_center <= STALL_SIDES[1])


IN_CONVERSATION_S = 30.0  # someone the robot answered this recently is still talking to it


def in_conversation(now: float, last_answer_at: float | None, quiet_s: float = IN_CONVERSATION_S) -> bool:
    """Whether the robot is talking with someone: then a person "walking up" is them coming back into view (the
    robot turned or walked away and back) and is not greeted again."""
    return last_answer_at is not None and now - last_answer_at < quiet_s


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
