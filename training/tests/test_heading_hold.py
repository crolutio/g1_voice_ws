"""Heading hold for the walk scene: walking straight keeps the direction the robot faced when it started."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from heading_hold import HeadingHold  # noqa: E402


def test_walking_straight_steers_back_to_the_starting_heading() -> None:
    hold = HeadingHold(gain=1.5, max_turn=0.4)
    assert hold.turn_command((0.5, 0.0, 0.0), yaw=0.0) == 0.0  # remembers facing 0
    drifted_left = hold.turn_command((0.5, 0.0, 0.0), yaw=0.1)
    assert drifted_left == pytest.approx(-0.15)  # turn right, back toward 0
    assert hold.turn_command((0.5, 0.0, 0.0), yaw=-1.0) == pytest.approx(0.4)  # limited


def test_a_commanded_turn_is_passed_through_and_sets_a_new_heading() -> None:
    hold = HeadingHold(gain=1.5, max_turn=0.4)
    hold.turn_command((0.5, 0.0, 0.0), yaw=0.0)
    assert hold.turn_command((0.0, 0.0, 0.8), yaw=0.3) == 0.8
    assert hold.turn_command((0.5, 0.0, 0.0), yaw=1.2) == 0.0  # straight again: holds the new heading


def test_no_steering_while_standing_and_angles_wrap_around() -> None:
    hold = HeadingHold(gain=1.5, max_turn=0.4)
    assert hold.turn_command((0.0, 0.0, 0.0), yaw=0.5) == 0.0
    hold.turn_command((0.5, 0.0, 0.0), yaw=math.pi - 0.05)
    across = hold.turn_command((0.5, 0.0, 0.0), yaw=-math.pi + 0.05)  # 0.1 rad past +pi, the short way
    assert across == pytest.approx(-0.15)
