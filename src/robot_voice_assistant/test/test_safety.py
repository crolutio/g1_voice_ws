"""Keeping a safe distance from people: walking, turning and gestures."""

import pytest

from robot_voice_assistant.motions import VelocityCommand
from robot_voice_assistant.safety import (
    GESTURE_CLEARANCE_M,
    PERSONAL_SPACE_M,
    TURN_CLEARANCE_M,
    gesture_allowed,
    room_to_walk,
    safe_command,
    turn_allowed,
)

# people in the robot's frame: x forward, y to the left (metres)
VISITOR_AT_STALL = [(1.5, 0.27)]


def test_the_robot_can_walk_up_to_a_metre_from_the_visitor_and_no_further() -> None:
    room = room_to_walk(VISITOR_AT_STALL, forward=True)
    assert room == pytest.approx(1.5 - (PERSONAL_SPACE_M ** 2 - 0.27 ** 2) ** 0.5)
    assert room_to_walk([(0.9, 0.0)], forward=True) == 0.0  # already closer than a metre: not a step closer


def test_people_beside_or_behind_the_path_do_not_stop_a_walk_forward() -> None:
    assert room_to_walk([(1.0, 1.5), (-2.0, 0.0)], forward=True) == float("inf")


def test_walking_backward_looks_behind_the_robot() -> None:
    assert room_to_walk([(-1.6, 0.0)], forward=False) == pytest.approx(0.6)
    assert room_to_walk(VISITOR_AT_STALL, forward=False) == float("inf")  # the visitor in front: free to back off


def test_a_walk_toward_someone_stops_while_there_is_still_room_to_coast() -> None:
    walking = VelocityCommand(forward=0.5)
    assert safe_command(walking, [(3.0, 0.0)])[1] is None  # far enough: keep walking
    command, reason = safe_command(walking, [(1.2, 0.0)])  # 0.2 m left before the personal space
    assert command.forward == 0.0 and "1 m" in reason
    backing, reason = safe_command(VelocityCommand(forward=-0.3), [(1.2, 0.0)])
    assert backing.forward == -0.3 and reason is None  # moving away from someone is always allowed


def test_turning_on_the_spot_needs_nobody_within_arm_s_reach() -> None:
    assert turn_allowed(VISITOR_AT_STALL)
    assert not turn_allowed([(TURN_CLEARANCE_M - 0.1, 0.0)])
    command, reason = safe_command(VelocityCommand(turn=0.5), [(0.5, 0.2)])
    assert command.turn == 0.0 and reason


def test_arm_gestures_need_clearance_and_small_ones_do_not() -> None:
    assert gesture_allowed("wave_right", VISITOR_AT_STALL)
    close = [(GESTURE_CLEARANCE_M - 0.1, 0.0)]
    for gesture in ("wave_right", "wave_left", "point_forward", "open_arms", "celebrate", "bow"):
        assert not gesture_allowed(gesture, close), gesture
    for gesture in ("nod", "shake", "none", "look_left"):
        assert gesture_allowed(gesture, close), gesture


def test_without_knowing_where_people_are_nothing_is_limited() -> None:
    assert room_to_walk([], forward=True) == float("inf") and turn_allowed([]) and gesture_allowed("wave_right", [])


def test_the_people_topic_is_read_and_bad_messages_mean_nobody_known() -> None:
    from robot_voice_assistant.safety import parse_people

    assert parse_people('[[1.5, 0.27], [-2.0, 1.0]]') == [(1.5, 0.27), (-2.0, 1.0)]
    assert parse_people("not json") == [] and parse_people('{"x": 1}') == []


def test_a_motion_that_would_bring_the_robot_too_close_does_not_start() -> None:
    from robot_voice_assistant.safety import motion_blocked

    right_in_front = [(1.1, 0.0)]  # 0.1 m left before the personal space
    assert "ahead" in motion_blocked("walk_forward", right_in_front)
    assert "ahead" in motion_blocked("keep_walking", right_in_front)
    assert motion_blocked("walk_backward", right_in_front) is None  # stepping away is fine
    assert motion_blocked("turn_left", right_in_front) is None  # 1.1 m: room for the arms
    assert "reach" in motion_blocked("turn_around", [(0.5, 0.0)])
    assert motion_blocked("stop", [(0.3, 0.0)]) is None  # stopping is always allowed
    assert motion_blocked("walk_forward", VISITOR_AT_STALL) is None  # 0.54 m of room: it walks, then stops at 1 m
