"""Reward maths for knee-bending walking. Run: ~/g1_voice_ws/.venv/bin/python -m pytest -q ~/g1_voice_ws/training/tests"""

import os
import sys

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
