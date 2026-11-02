"""Leg kinematics and foot-placement IK for the Unitree R1. Run with the workspace Python:

    ~/g1_voice_ws/.venv/bin/python -m pytest -q ~/g1_voice_ws/training/tests
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from foot_placement.leg_kinematics import LegKinematics, rotation_error  # noqa: E402

URDF = os.path.expanduser("~/g1_voice_ws/robots/r1/R1.urdf")


@pytest.fixture(scope="module")
def legs() -> dict[str, LegKinematics]:
    return {side: LegKinematics.from_urdf(URDF, side) for side in ("left", "right")}


def test_each_leg_has_six_joints_from_hip_to_ankle(legs: dict[str, LegKinematics]) -> None:
    assert legs["left"].joint_names == [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    ]
    assert len(legs["right"].joint_names) == 6


def test_the_legs_are_mirror_images(legs: dict[str, LegKinematics]) -> None:
    zero = torch.zeros(1, 6)
    left, _ = legs["left"].foot_pose(zero)
    right, _ = legs["right"].foot_pose(zero)
    assert left[0, 1] == pytest.approx(-right[0, 1].item(), abs=1e-4)  # same distance either side
    assert left[0, 0] == pytest.approx(right[0, 0].item(), abs=1e-4)
    assert left[0, 2] == pytest.approx(right[0, 2].item(), abs=1e-4)
    assert left[0, 2] < -0.5  # the ankle hangs well below the pelvis


def test_bending_the_knee_raises_the_foot(legs: dict[str, LegKinematics]) -> None:
    straight, bent = torch.zeros(1, 6), torch.zeros(1, 6)
    bent[0, 3] = 1.0
    assert legs["left"].foot_pose(bent)[0][0, 2] > legs["left"].foot_pose(straight)[0][0, 2] + 0.05


# The range of leg poses walking uses (hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll), in rad
WALKING_LOW = torch.tensor([-0.75, -0.25, -0.25, 0.05, -0.6, -0.2])
WALKING_HIGH = torch.tensor([0.25, 0.25, 0.25, 1.3, 0.1, 0.2])


def test_ik_reaches_walking_poses(legs: dict[str, LegKinematics]) -> None:
    torch.manual_seed(0)
    leg = legs["left"]
    target_angles = WALKING_LOW + (WALKING_HIGH - WALKING_LOW) * torch.rand(256, 6)
    target_position, target_rotation = leg.foot_pose(target_angles)
    start = leg.default_angles().expand(256, 6)
    solution = leg.solve(target_position, target_rotation, start, iterations=40)
    position, rotation = leg.foot_pose(solution)
    assert torch.max(torch.linalg.norm(position - target_position, dim=1)) < 1e-3  # within a millimetre
    assert torch.max(torch.linalg.norm(rotation_error(rotation, target_rotation), dim=1)) < torch.deg2rad(torch.tensor(0.5))


def test_ik_keeps_the_joints_inside_their_limits(legs: dict[str, LegKinematics]) -> None:
    leg = legs["right"]
    far_away = torch.tensor([[0.0, -0.1, -0.2]])  # unreachable: far too high
    solution = leg.solve(far_away, torch.eye(3).unsqueeze(0), leg.default_angles(), iterations=20)
    assert torch.all(solution >= leg.lower_limits - 1e-6) and torch.all(solution <= leg.upper_limits + 1e-6)


# --- safety while walking: an unreachable target (a big catch step, a stumble) must never flip the leg ---


def test_unreachable_targets_never_flip_the_leg_into_a_strange_pose(legs: dict[str, LegKinematics]) -> None:
    from foot_placement.leg_kinematics import WALKING_RANGE

    leg = legs["left"]
    unreachable = torch.tensor([[0.6, 0.1, -0.3], [0.0, 0.1, -1.2], [-0.5, 0.4, -0.4], [0.0, 0.1, 0.1]])
    angles = leg.default_angles().expand(4, 6)
    for _ in range(50):  # warm-started step after step, as in the walk
        angles = leg.solve(unreachable, torch.eye(3).expand(4, 3, 3), angles, iterations=4, safe=True)
    assert torch.isfinite(angles).all()
    low = torch.tensor([WALKING_RANGE[name][0] for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee")])
    high = torch.tensor([WALKING_RANGE[name][1] for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee")])
    assert torch.all(angles[:, :4] >= low - 1e-6) and torch.all(angles[:, :4] <= high + 1e-6)
    # it still reaches as far as it can toward each target (the foot is on the way, not somewhere else)
    position, _ = leg.foot_pose(angles)
    hip = leg.hip_position()
    direction = torch.nn.functional.cosine_similarity(position - hip, unreachable - hip, dim=1)
    assert torch.all(direction[:3] > 0.85)
    assert torch.all(angles[:2, 3] < 0.6)  # stretched toward a far target, not folded up


def test_a_safe_solve_moves_each_joint_at_most_the_step_limit(legs: dict[str, LegKinematics]) -> None:
    from foot_placement.leg_kinematics import MAX_JOINT_STEP_RAD

    leg = legs["left"]
    far = torch.tensor([[0.4, 0.1, -0.5]])
    start = leg.default_angles()
    moved = leg.solve(far, torch.eye(3).unsqueeze(0), start, iterations=10, safe=True)
    assert torch.max(torch.abs(moved - start)) <= MAX_JOINT_STEP_RAD + 1e-6


def test_safe_mode_still_reaches_walking_poses(legs: dict[str, LegKinematics]) -> None:
    torch.manual_seed(1)
    leg = legs["right"]
    target_angles = WALKING_LOW + (WALKING_HIGH - WALKING_LOW) * torch.rand(64, 6)
    target_angles[:, 1:3] *= -1  # the right leg's roll and yaw are mirrored
    target_position, target_rotation = leg.foot_pose(target_angles)
    angles = leg.default_angles().expand(64, 6)
    for _ in range(10):
        angles = leg.solve(target_position, target_rotation, angles, iterations=4, safe=True)
    position, _ = leg.foot_pose(angles)
    assert torch.max(torch.linalg.norm(position - target_position, dim=1)) < 2e-3
