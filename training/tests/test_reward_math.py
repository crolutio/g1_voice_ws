"""Reward maths for knee-bending walking. Run: ~/g1_voice_ws/.venv/bin/python -m pytest -q ~/g1_voice_ws/training/tests"""

import os
import sys

import math

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "walk_task"))
from reward_math import knee_straightness, swing_foot_height_shortfall  # noqa: E402


def test_bent_knees_are_not_penalised() -> None:
    knees = torch.tensor([[0.5, 0.8], [0.2, 1.4]])
    assert torch.equal(knee_straightness(knees, min_bend=0.2), torch.zeros(2))


def test_knees_straighter_than_the_minimum_are_penalised_by_how_far() -> None:
    knees = torch.tensor([[-0.17, 0.5], [0.1, 0.0]])  # locked on the stop; slightly straight
    assert knee_straightness(knees, min_bend=0.2) == pytest.approx(torch.tensor([0.37, 0.3]), abs=1e-6)


def test_a_swinging_foot_lifted_high_enough_is_not_penalised() -> None:
    foot_z = torch.tensor([[0.05, 0.14]])  # left on the ground, right lifted 9 cm above it
    in_contact = torch.tensor([[True, False]])
    assert swing_foot_height_shortfall(foot_z, in_contact, target=0.08).item() == 0.0


def test_a_shuffling_foot_is_penalised_by_the_squared_shortfall() -> None:
    foot_z = torch.tensor([[0.05, 0.07]])  # right foot only 2 cm up: 6 cm short of 8 cm
    in_contact = torch.tensor([[True, False]])
    assert swing_foot_height_shortfall(foot_z, in_contact, target=0.08).item() == pytest.approx(0.06**2)


def test_feet_on_the_ground_or_both_in_the_air_are_not_judged() -> None:
    foot_z = torch.tensor([[0.05, 0.05], [0.20, 0.21]])
    in_contact = torch.tensor([[True, True], [False, False]])
    assert torch.equal(swing_foot_height_shortfall(foot_z, in_contact, target=0.08), torch.zeros(2))


def test_standing_still_when_told_to_stop_is_not_penalised() -> None:
    from reward_math import drift_while_standing

    velocity = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])  # vx, vy, wz
    command = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])  # the second robot is told to walk
    assert torch.equal(drift_while_standing(velocity, command), torch.zeros(2))


def test_drifting_after_a_stop_is_penalised_by_the_squared_speed() -> None:
    from reward_math import drift_while_standing

    velocity = torch.tensor([[0.25, 0.0, 0.0], [0.0, 0.1, 0.4]])
    command = torch.zeros(2, 3)
    assert drift_while_standing(velocity, command) == pytest.approx(torch.tensor([0.0625, 0.01 + 0.5 * 0.16]))


def test_a_turn_command_is_not_standing_still() -> None:
    from reward_math import drift_while_standing

    velocity = torch.tensor([[0.0, 0.0, 0.8]])
    command = torch.tensor([[0.0, 0.0, 0.8]])
    assert drift_while_standing(velocity, command).item() == 0.0


def test_both_feet_in_the_air_counts_as_a_jump() -> None:
    from reward_math import both_feet_in_air

    in_contact = torch.tensor([[False, False], [True, False], [True, True]])
    assert torch.equal(both_feet_in_air(in_contact), torch.tensor([1.0, 0.0, 0.0]))


def test_desired_stance_alternates_with_a_short_double_support() -> None:
    from reward_math import desired_stance

    phase = torch.tensor([0.0, 0.3, 0.55, 0.7, 0.95])
    stance = desired_stance(phase, stance_fraction=0.6)
    # left foot down for the first 60% of the cycle, right foot from 50% on (and wrapping to 10%)
    assert stance.tolist() == [[True, True], [True, False], [True, True], [False, True], [False, True]]


def test_feet_following_the_step_clock_are_rewarded() -> None:
    from reward_math import gait_phase_match

    phase = torch.tensor([0.3, 0.3, 0.3])
    in_contact = torch.tensor([[True, False], [True, True], [False, False]])  # right, half right, wrong
    moving = torch.tensor([True, True, True])
    assert gait_phase_match(in_contact, phase, moving, 0.6).tolist() == [1.0, 0.5, 0.5]


def test_standing_robots_should_keep_both_feet_down() -> None:
    from reward_math import gait_phase_match

    phase = torch.tensor([0.3, 0.3])
    in_contact = torch.tensor([[True, True], [True, False]])
    moving = torch.tensor([False, False])
    assert gait_phase_match(in_contact, phase, moving, 0.6).tolist() == [1.0, 0.5]


def test_one_foot_ahead_of_the_other_is_rewarded_while_walking() -> None:
    from reward_math import stride_length

    in_contact = torch.tensor([[True, True], [True, True], [True, False], [True, True]])
    feet_x = torch.tensor([[0.15, -0.10], [0.01, -0.01], [0.3, -0.3], [0.5, -0.5]])  # along the heading
    walking = torch.tensor([True, True, True, False])
    lengths = stride_length(in_contact, feet_x, walking, max_length=0.3)
    assert lengths.tolist() == pytest.approx([0.25, 0.02, 0.0, 0.0])  # capped at 0.3; only with both feet down


# --- the human gait: swing with a bent knee, straighten before landing, land in front, heel first ---


def test_a_swinging_knee_bent_enough_is_not_penalised() -> None:
    from reward_math import swing_knee_shortfall

    knee = torch.tensor([[0.2, 0.8], [0.2, 0.3]])
    in_contact = torch.tensor([[True, False], [True, False]])  # the right foot swings
    moving = torch.tensor([True, True])
    assert swing_knee_shortfall(knee, in_contact, moving, target=0.5).tolist() == pytest.approx([0.0, 0.2])


def test_standing_knees_are_not_judged_by_the_swing_rule() -> None:
    from reward_math import swing_knee_shortfall

    knee = torch.tensor([[0.05, 0.05]])
    assert swing_knee_shortfall(knee, torch.tensor([[True, True]]), torch.tensor([True]), 0.5).item() == 0.0
    assert swing_knee_shortfall(knee, torch.tensor([[True, False]]), torch.tensor([False]), 0.5).item() == 0.0


def test_a_foot_landing_with_a_bent_knee_is_penalised() -> None:
    from reward_math import landing_knee_excess

    knee = torch.tensor([[0.1, 0.6], [0.1, 0.6]])
    first_contact = torch.tensor([[True, False], [False, True]])  # landing: left (straight) / right (bent)
    assert landing_knee_excess(knee, first_contact, max_bend=0.25).tolist() == pytest.approx([0.0, 0.35])


def test_the_landing_foot_should_be_in_front_when_walking_forward_and_behind_when_walking_back() -> None:
    from reward_math import landing_in_front

    feet_x = torch.tensor([[0.2, -0.1], [0.2, -0.1], [0.2, -0.1]])  # relative to the hips, along the heading
    first_contact = torch.tensor([[True, False], [True, False], [False, True]])
    direction = torch.tensor([1.0, -1.0, -1.0])  # forward, backward, backward
    assert landing_in_front(feet_x, first_contact, direction, max_distance=0.15).tolist() == pytest.approx(
        [0.15, 0.0, 0.1]
    )


def test_landing_heel_first_is_rewarded() -> None:
    from reward_math import heel_first

    toe_up = torch.tensor([[0.1, -0.05], [0.3, 0.0]])  # sine of the toe-up angle of each foot
    first_contact = torch.tensor([[True, True], [True, False]])
    assert heel_first(toe_up, first_contact, max_tilt=0.15).tolist() == pytest.approx([0.1, 0.15])


def test_a_standing_leg_with_a_deeply_bent_knee_is_penalised() -> None:
    from reward_math import stance_knee_excess

    knee = torch.tensor([[1.2, 0.3], [0.3, 1.2]])
    in_contact = torch.tensor([[True, False], [True, False]])  # only the left leg stands
    assert stance_knee_excess(knee, in_contact, max_bend=0.6).tolist() == pytest.approx([0.6, 0.0])


# --- closer legs, and each foot steps past the other ---


def test_feet_too_wide_or_too_close_are_penalised() -> None:
    from reward_math import step_width_error

    feet_y = torch.tensor([[0.11, -0.11], [0.25, -0.25], [0.05, -0.05]])  # 22 cm, 50 cm, 10 cm apart
    assert step_width_error(feet_y, min_width=0.15, max_width=0.25).tolist() == pytest.approx([0.0, 0.25, 0.05])


def test_a_landing_foot_that_passes_the_other_foot_is_rewarded() -> None:
    from reward_math import landing_passes_other_foot

    feet_x = torch.tensor([[0.20, -0.05], [0.20, -0.05], [-0.05, 0.20]])
    first_contact = torch.tensor([[True, False], [False, True], [True, False]])
    direction = torch.tensor([1.0, 1.0, 1.0])  # walking forward
    # left lands 25 cm ahead of the right: good; right lands 25 cm behind: nothing; left lands behind: nothing
    assert landing_passes_other_foot(feet_x, first_contact, direction, max_distance=0.3).tolist() == pytest.approx(
        [0.25, 0.0, 0.0]
    )


def test_walking_backward_the_landing_foot_should_pass_behind_the_other() -> None:
    from reward_math import landing_passes_other_foot

    feet_x = torch.tensor([[-0.20, 0.05]])
    first_contact = torch.tensor([[True, False]])
    assert landing_passes_other_foot(feet_x, first_contact, torch.tensor([-1.0]), 0.3).item() == pytest.approx(0.25)


# --- quaternions in Isaac Lab's (x, y, z, w) order ---


def _xyzw(axis: tuple[float, float, float], angle: float) -> torch.Tensor:
    import math

    s = math.sin(angle / 2)
    return torch.tensor([[axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2)]])


def test_heading_comes_from_an_xyzw_quaternion() -> None:
    from reward_math import heading_from_quat_xyzw

    for yaw in (0.0, 0.5, 2.0, -2.5):
        assert heading_from_quat_xyzw(_xyzw((0, 0, 1), yaw)).item() == pytest.approx(yaw, abs=1e-5)


def test_toes_up_is_positive_for_a_foot_pitched_heel_down() -> None:
    from reward_math import toe_up_from_quat_xyzw

    # a rotation about +y by a negative angle lifts the x (toe) axis
    assert toe_up_from_quat_xyzw(_xyzw((0, 1, 0), -0.2)).item() == pytest.approx(math.sin(0.2), abs=1e-5)
    assert toe_up_from_quat_xyzw(_xyzw((0, 1, 0), 0.2)).item() == pytest.approx(-math.sin(0.2), abs=1e-5)
    assert toe_up_from_quat_xyzw(_xyzw((0, 0, 1), 1.0)).item() == pytest.approx(0.0, abs=1e-6)


# --- balance: the capture point (where the robot would have to step to stop) must stay sideways between the feet ---


def test_capture_point_between_the_feet_is_not_penalised() -> None:
    from reward_math import sideways_capture_point_outside

    feet_y = torch.tensor([[0.10, -0.10]])  # heading frame, relative to the pelvis
    still = sideways_capture_point_outside(torch.tensor([0.0]), torch.tensor([0.7]), feet_y, margin=0.03)
    assert still.item() == 0.0
    swaying = sideways_capture_point_outside(torch.tensor([0.2]), torch.tensor([0.7]), feet_y, margin=0.03)
    assert swaying.item() == 0.0  # moving 0.2 m/s sideways: still lands inside the feet


def test_capture_point_past_the_outer_foot_is_penalised_by_how_far() -> None:
    from reward_math import sideways_capture_point_outside

    feet_y = torch.tensor([[0.10, -0.10], [0.10, -0.10]])
    omega = math.sqrt(9.81 / 0.7)
    falling_left = torch.tensor([0.6, -0.9])  # m/s sideways: toward the left foot, and fast to the right
    outside = sideways_capture_point_outside(falling_left, torch.tensor([0.7, 0.7]), feet_y, margin=0.03)
    assert outside[0].item() == pytest.approx(0.6 / omega - 0.13, abs=1e-5)
    assert outside[1].item() == pytest.approx(0.9 / omega - 0.13, abs=1e-5)
