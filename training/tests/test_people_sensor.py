"""Where the people around the robot are, as its range sensor reports them, and their physics bodies."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from people_sensor import body_positions, robot_frame  # noqa: E402


def test_people_are_reported_in_the_robot_s_frame() -> None:
    # the robot at (10, 5) facing +y: someone 1.5 m further along +y is straight ahead, someone at +x is on its right
    people = robot_frame([(10.0, 6.5), (11.0, 5.0)], robot_xy=(10.0, 5.0), robot_yaw=math.pi / 2)
    assert people[0] == pytest.approx((1.5, 0.0), abs=1e-9)
    assert people[1] == pytest.approx((0.0, -1.0), abs=1e-9)


def test_people_out_of_range_or_not_there_are_left_out() -> None:
    assert robot_frame([(20.0, 0.0), None, (1.0, 0.0)], robot_xy=(0.0, 0.0), robot_yaw=0.0) == [(1.0, 0.0)]


def test_people_who_are_not_there_have_their_bodies_parked_far_away() -> None:
    positions = body_positions([(1.5, 0.27), None], height_m=1.74)
    assert positions[0] == pytest.approx((1.5, 0.27, 0.87))
    assert math.hypot(*positions[1][:2]) > 100  # out of the hall, so nothing bumps into it
    assert body_positions([None, None], 1.74)[0] != body_positions([None, None], 1.74)[1]  # not on top of each other
