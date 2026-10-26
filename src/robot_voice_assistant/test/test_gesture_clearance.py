"""The arms must never hit the head or the torso, in any gesture or when switching between gestures.

Self-collisions are disabled in the simulation, so a bad pose shows up as an arm passing through the
head. These tests use forward kinematics of the real R1 (Unitree URDF and meshes).
"""

import numpy as np
import pytest

from robot_voice_assistant.gestures import GESTURE_KEYFRAMES, NEUTRAL_POSE, GestureTrajectory, Pose
from robot_voice_assistant.kinematics import HEAD_LINK, TORSO_LINK, RobotKinematics

MIN_CLEARANCE_M = {HEAD_LINK: 0.04, TORSO_LINK: 0.03}
KINEMATICS = RobotKinematics.load()


def _closest_approach(trajectory: GestureTrajectory, step_s: float) -> tuple[float, str, float]:
    """(margin above the required clearance, what was closest, when) over the whole trajectory."""
    worst = (np.inf, "", 0.0)
    for t in np.arange(0.0, trajectory.duration_s + 1e-9, step_s):
        for (side, body), distance in KINEMATICS.clearances(trajectory.sample(float(t))).items():
            margin = distance - MIN_CLEARANCE_M[body]
            if margin < worst[0]:
                worst = (margin, f"{side} arm vs {body} ({distance * 100:.1f} cm)", float(t))
    return worst


def test_neutral_pose_is_clear() -> None:
    clearances = KINEMATICS.clearances(NEUTRAL_POSE)
    assert all(distance > MIN_CLEARANCE_M[body] for (_, body), distance in clearances.items())


@pytest.mark.parametrize("gesture", sorted(GESTURE_KEYFRAMES))
def test_gesture_keeps_hands_away_from_head_and_body(gesture: str) -> None:
    margin, what, when = _closest_approach(GestureTrajectory(gesture, NEUTRAL_POSE), step_s=0.04)
    assert margin >= 0.0, f"{gesture}: {what} at t={when:.2f}s"


def _interrupt_poses(gesture: str) -> list[Pose]:
    """Poses a gesture can be interrupted in: its first keyframe and its middle keyframe."""
    keyframes = GESTURE_KEYFRAMES[gesture]
    if not keyframes:
        return []
    return [keyframes[0][1], keyframes[len(keyframes) // 2][1]]


@pytest.mark.parametrize("interrupted", sorted(g for g in GESTURE_KEYFRAMES if g != "none"))
def test_switching_to_another_gesture_mid_way_is_clear(interrupted: str) -> None:
    for start_pose in _interrupt_poses(interrupted):
        for gesture in GESTURE_KEYFRAMES:
            margin, what, when = _closest_approach(GestureTrajectory(gesture, start_pose), step_s=0.1)
            assert margin >= 0.0, f"{interrupted} -> {gesture}: {what} at t={when:.2f}s"
