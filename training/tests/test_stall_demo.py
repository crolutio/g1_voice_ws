"""Scoring the robot's answers in the stall demo."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stall_demo import score_answer, summarise  # noqa: E402


def test_an_answer_naming_the_object_is_right() -> None:
    assert score_answer("You're holding an apple.", "apple") == "right"
    assert score_answer("That looks like a mustard bottle!", "bottle") == "right"
    assert score_answer("It's a red cup.", "mug") == "right"  # a mug is a cup
    assert score_answer("A pair of scissors.", "scissors") == "right"


def test_naming_something_else_or_nothing_is_wrong_or_unsure() -> None:
    assert score_answer("That's a banana.", "apple") == "wrong"
    assert score_answer("I can't see it clearly, could you hold it closer?", "mug") == "not seen"
    assert score_answer("Either an apple or a banana.", "apple") == "wrong"  # hedging between two is not an answer


def test_the_summary_counts_per_object() -> None:
    trials = [
        {"object": "apple", "greeted": True, "result": "right", "answer_s": 1.5},
        {"object": "apple", "greeted": True, "result": "wrong", "answer_s": 2.5},
        {"object": "mug", "greeted": False, "result": "right", "answer_s": 1.0},
    ]
    summary = summarise(trials)
    assert summary["apple"] == {"trials": 2, "greeted": 2, "right": 1, "wrong": 1, "not seen": 0}
    assert summary["total"]["right"] == 2 and summary["total"]["greeted"] == 2 and summary["total"]["trials"] == 3


def test_a_later_session_adds_to_the_report_of_the_earlier_ones(tmp_path) -> None:
    from stall_demo import load_trials, write_report

    assert load_trials(str(tmp_path)) == []
    first = [{"visit": "1_apple", "object": "apple", "greeted": True, "result": "right", "answer": "An apple.",
              "scene": "a person, holding an apple", "answer_s": 1.2}]
    write_report(first, str(tmp_path))
    assert load_trials(str(tmp_path)) == first
