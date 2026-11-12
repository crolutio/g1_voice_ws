"""Scoring the end-to-end visitor conversation: motions measured on the odometry, gestures, replies, capacity."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from visitor_conversation import (TURNS, Turn, capacity, check_turn, motion_delta)  # noqa: E402


def test_a_motion_is_measured_in_the_robot_s_own_frame() -> None:
    # facing +y: one metre along +y is one metre forward; a quarter turn left is +90 degrees
    forward, turned = motion_delta((0.0, 0.0, math.pi / 2), (0.0, 1.0, math.pi))
    assert forward == pytest.approx(1.0) and turned == pytest.approx(90.0)
    _, back_round = motion_delta((0.0, 0.0, math.radians(170)), (0.0, 0.0, math.radians(-170)))
    assert back_round == pytest.approx(20.0)  # across +-180 degrees


def test_a_turn_passes_when_the_robot_did_what_it_was_asked() -> None:
    turn = Turn("Please turn left.", motion="turn_left")
    ok = check_turn(turn, reply="Turning left now.", gestures=["nod"], motions=["turn_left"], moved=(0.02, 84.0))
    assert ok == []
    wrong = check_turn(turn, reply="Okay.", gestures=["nod"], motions=["turn_right"], moved=(0.0, -88.0))
    assert any("turn_left" in problem for problem in wrong) and any("turned" in problem for problem in wrong)


def test_a_turn_that_asks_a_gesture_or_a_fact_is_checked_on_the_reply_and_the_gesture() -> None:
    wave = Turn("Can you wave at me?", gestures={"wave_right", "wave_left"})
    assert check_turn(wave, reply="Hello!", gestures=["wave_right"], motions=[], moved=None) == []
    assert check_turn(wave, reply="Hello!", gestures=["nod"], motions=[], moved=None) != []
    name = Turn("What's your name?", says=["R1"])
    assert check_turn(name, reply="I'm R1, nice to meet you.", gestures=[], motions=[], moved=None) == []
    assert check_turn(name, reply="Nice to meet you.", gestures=[], motions=[], moved=None) != []
    hand = Turn("Can you tell me what is in my hand?", holds="apple")
    assert check_turn(hand, reply="You are holding an apple.", gestures=[], motions=[], moved=None) == []
    assert check_turn(hand, reply="It's a banana.", gestures=[], motions=[], moved=None) != []


def test_no_answer_is_a_failure() -> None:
    assert check_turn(Turn("Hi"), reply=None, gestures=[], motions=[], moved=None) == ["no reply"]


def test_capacity_follows_the_tightest_groq_limit() -> None:
    # 15 requests and 30,000 tokens per conversation; 1,000 requests and 200,000 tokens a day
    days = capacity(requests=15, tokens=30_000, turns=15, minutes=6.0, rpd=1000, tpd=200_000, rpm=30, tpm=8000)
    assert days["conversations_per_day"] == pytest.approx(6.67, abs=0.01)  # tokens run out first
    assert days["limited_by"] == "tokens per day"
    assert days["tokens_per_turn"] == 2000 and days["turns_per_minute_at_tpm"] == pytest.approx(4.0)


def test_the_conversation_covers_everything_a_visitor_asks_for() -> None:
    motions = {turn.motion for turn in TURNS if turn.motion}
    assert {"turn_left", "turn_right", "walk_forward", "walk_backward", "turn_around", "keep_walking",
            "stop"} <= motions
    assert any(turn.holds for turn in TURNS) and any(turn.about_the_view for turn in TURNS)
    assert sum(1 for turn in TURNS if turn.gestures) >= 3


def test_a_motion_that_was_not_measured_does_not_pass() -> None:
    turn = Turn("Come here.", motion="walk_forward")
    assert check_turn(turn, reply="On my way!", gestures=[], motions=["walk_forward"], moved=None) == [
        "the motion was not measured on the odometry"]
