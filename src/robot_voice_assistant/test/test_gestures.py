import math

import pytest

from robot_voice_assistant.gestures import (
    GESTURE_DESCRIPTIONS,
    GESTURE_KEYFRAMES,
    JOINT_LIMITS_DEG,
    NEUTRAL_POSE,
    UPPER_BODY_JOINTS,
    GestureTrajectory,
)


def test_every_described_gesture_has_keyframes() -> None:
    assert set(GESTURE_DESCRIPTIONS) == set(GESTURE_KEYFRAMES)
    assert "none" in GESTURE_DESCRIPTIONS


def test_neutral_pose_covers_all_controlled_joints() -> None:
    assert set(NEUTRAL_POSE) == set(UPPER_BODY_JOINTS)


@pytest.mark.parametrize("gesture", sorted(GESTURE_KEYFRAMES))
def test_keyframes_respect_joint_limits(gesture: str) -> None:
    poses = [NEUTRAL_POSE] + [pose for _, pose in GESTURE_KEYFRAMES[gesture]]
    for pose in poses:
        for joint, angle in pose.items():
            assert joint in UPPER_BODY_JOINTS, f"{gesture}: unknown joint {joint}"
            lower, upper = JOINT_LIMITS_DEG[joint]
            # Keep a 2 degree safety margin from the hard limits.
            assert math.radians(lower + 2) <= angle <= math.radians(upper - 2), f"{gesture}: {joint}={angle}"


@pytest.mark.parametrize("gesture", sorted(GESTURE_KEYFRAMES))
def test_keyframe_times_increase(gesture: str) -> None:
    times = [t for t, _ in GESTURE_KEYFRAMES[gesture]]
    assert times == sorted(times)
    assert all(t > 0 for t in times)


def test_trajectory_starts_at_current_pose_and_ends_at_neutral() -> None:
    start = {joint: angle + 0.1 for joint, angle in NEUTRAL_POSE.items()}
    trajectory = GestureTrajectory("wave_right", start)
    assert trajectory.sample(0.0) == pytest.approx(start)
    assert trajectory.sample(trajectory.duration_s) == pytest.approx(NEUTRAL_POSE)
    assert trajectory.sample(trajectory.duration_s + 5.0) == pytest.approx(NEUTRAL_POSE)
    assert trajectory.is_finished(trajectory.duration_s + 0.01)


def test_trajectory_passes_through_keyframes() -> None:
    trajectory = GestureTrajectory("wave_right", NEUTRAL_POSE)
    keyframe_time, keyframe_pose = GESTURE_KEYFRAMES["wave_right"][0]
    sample = trajectory.sample(keyframe_time)
    for joint, angle in keyframe_pose.items():
        assert sample[joint] == pytest.approx(angle)


def test_trajectory_is_continuous() -> None:
    trajectory = GestureTrajectory("celebrate", NEUTRAL_POSE)
    step = 0.02
    previous = trajectory.sample(0.0)
    t = step
    while t <= trajectory.duration_s:
        current = trajectory.sample(t)
        assert max(abs(current[j] - previous[j]) for j in current) < 0.25
        previous = current
        t += step


def test_none_gesture_returns_to_neutral() -> None:
    start = {joint: angle + 0.2 for joint, angle in NEUTRAL_POSE.items()}
    trajectory = GestureTrajectory("none", start)
    assert trajectory.sample(trajectory.duration_s) == pytest.approx(NEUTRAL_POSE)


def test_unknown_gesture_is_rejected() -> None:
    with pytest.raises(KeyError):
        GestureTrajectory("backflip", NEUTRAL_POSE)
