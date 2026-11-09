"""GITEX visitors: walking a looping route, facing where they go, feet in step with the ground, pausing."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from visitors import Visitor, Waypoint  # noqa: E402

FRAMES = 80
CYCLE_M = 1.6  # metres covered by one walk cycle


def walker(route: list[Waypoint], start_s: float = 0.0) -> Visitor:
    return Visitor(route, speed=0.8, cycle_m=CYCLE_M, frames=FRAMES, standing_frame=3, start_s=start_s)


def test_a_visitor_walks_the_route_at_its_speed_and_faces_the_way() -> None:
    v = walker([Waypoint(0, 0), Waypoint(8, 0), Waypoint(8, 6)])
    x, y, yaw, _ = v.pose(5.0)  # 4 m along the first leg
    assert (x, y) == pytest.approx((4.0, 0.0))
    assert yaw == pytest.approx(0.0)
    x, y, yaw, _ = v.pose(12.5)  # 8 m on the first leg, then 2 m up the second
    assert (x, y) == pytest.approx((8.0, 2.0))
    assert yaw == pytest.approx(math.pi / 2)


def test_the_route_loops_back_to_the_start() -> None:
    v = walker([Waypoint(0, 0), Waypoint(4, 0)])  # out 4 m and back 4 m: 8 m, 10 s a loop
    assert v.pose(10.0)[:2] == pytest.approx((0.0, 0.0))
    x, y, yaw, _ = v.pose(7.5)  # on the way back
    assert (x, y) == pytest.approx((2.0, 0.0)) and abs(yaw) == pytest.approx(math.pi)


def test_the_walk_cycle_matches_the_distance_so_feet_do_not_slide() -> None:
    v = walker([Waypoint(0, 0), Waypoint(100, 0)])
    assert v.pose(0.0)[3] == 0
    seconds_per_cycle = CYCLE_M / 0.8
    assert v.pose(seconds_per_cycle / 4)[3] == FRAMES // 4
    assert v.pose(seconds_per_cycle)[3] == 0  # one full cycle per CYCLE_M walked


def test_a_visitor_pauses_standing_at_a_waypoint() -> None:
    v = walker([Waypoint(0, 0), Waypoint(4, 0, pause_s=3.0), Waypoint(4, 4)])
    x, y, _, frame = v.pose(6.5)  # arrived after 5 s, pausing until 8 s
    assert (x, y) == pytest.approx((4.0, 0.0)) and frame == 3  # the standing frame
    assert v.pose(9.0)[:2] == pytest.approx((4.0, 0.8))  # walking again


def test_visitors_can_start_part_way_along_their_route() -> None:
    v = walker([Waypoint(0, 0), Waypoint(8, 0)], start_s=2.5)
    assert v.pose(0.0)[:2] == pytest.approx((2.0, 0.0))
