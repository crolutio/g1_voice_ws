"""The live stall demo's rounds: a visitor with a random object, the conversation, goodbye, the next visitor."""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from live_stall import GOODBYE_S, PAUSE_S, StallRounds, pick_object  # noqa: E402
from stall_demo import OBJECTS  # noqa: E402


def test_each_visitor_brings_a_different_object_from_the_last() -> None:
    rng = random.Random(1)
    previous = None
    for _ in range(50):
        held = pick_object(previous, rng)
        assert held in OBJECTS and held != previous
        previous = held


def test_a_round_goes_from_arrival_through_the_conversation_to_goodbye() -> None:
    rounds = StallRounds(random.Random(2))
    action = rounds.update(0.0, "idle", in_conversation=False)
    assert action[0] == "new_round" and action[1] in OBJECTS
    held = action[1]
    assert rounds.update(1.0, f"approaching {held}", False) is None
    assert rounds.update(5.0, f"at_stall {held}", False) is None
    assert rounds.update(6.0, f"at_stall {held}", True) is None  # greeted: the conversation is on
    assert rounds.update(60.0, f"at_stall {held}", True) is None  # talking
    assert rounds.update(61.0, f"at_stall {held}", False) is None  # goodbye said: the robot answers it first
    assert rounds.update(61.0 + GOODBYE_S, f"at_stall {held}", False) == ("leave",)
    assert rounds.update(70.0, f"leaving {held}", False) is None
    assert rounds.update(75.0, "idle", False) is None  # gone: a short pause
    nxt = rounds.update(75.0 + PAUSE_S, "idle", False)
    assert nxt[0] == "new_round" and nxt[1] != held and rounds.round == 2


def test_a_visitor_who_leaves_by_themselves_also_starts_a_new_round() -> None:
    rounds = StallRounds(random.Random(3))
    held = rounds.update(0.0, "idle", False)[1]
    rounds.update(5.0, f"at_stall {held}", False)
    assert rounds.update(700.0, "idle", False) is None  # left after their longest stay, nobody said goodbye
    assert rounds.update(700.0 + PAUSE_S, "idle", False)[0] == "new_round"
