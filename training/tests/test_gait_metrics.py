"""Gait measurements. Run: ~/g1_voice_ws/.venv/bin/python -m pytest -q ~/g1_voice_ws/training/tests"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gait_metrics import gait_summary  # noqa: E402


def walking(steps: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Left and right foot alternate with short double-support phases; left foot ahead, then right."""
    phase = np.arange(steps) % 40
    left = phase < 24  # stance 0-23
    right = (phase >= 18) | (phase < 4)  # stance 18-39 and 0-3: overlaps = double support
    contact = np.stack([left, right], axis=1)
    foot_x = np.stack([np.where(phase < 20, 0.12, -0.12), np.where(phase < 20, -0.12, 0.12)], axis=1)
    return contact, foot_x


def test_walking_has_no_flight_and_alternating_steps() -> None:
    contact, foot_x = walking()
    summary = gait_summary(contact, foot_x, dt=0.02)
    assert summary["flight"] == 0.0
    assert summary["double_support"] == pytest.approx(10 / 40)
    assert summary["single_stance"] == pytest.approx(30 / 40)
    assert summary["step_length"] == pytest.approx(0.24)  # feet 24 cm apart, front to back
    assert summary["steps_per_s"] == pytest.approx(2 * 200 / 40 / (200 * 0.02), rel=0.15)


def test_jumping_shows_as_flight() -> None:
    phase = np.arange(200) % 20
    on_ground = phase < 12  # both feet land and take off together
    contact = np.stack([on_ground, on_ground], axis=1)
    summary = gait_summary(contact, np.zeros((200, 2)), dt=0.02)
    assert summary["flight"] == pytest.approx(8 / 20)
    assert summary["single_stance"] == 0.0
    assert summary["step_length"] == pytest.approx(0.0)


def test_standing_is_all_double_support() -> None:
    contact = np.ones((100, 2), dtype=bool)
    summary = gait_summary(contact, np.zeros((100, 2)), dt=0.02)
    assert summary["double_support"] == 1.0 and summary["steps_per_s"] == 0.0


def test_step_width_is_the_side_to_side_distance_between_the_feet() -> None:
    contact, foot_x = walking()
    foot_y = np.tile([0.11, -0.11], (len(contact), 1))
    assert gait_summary(contact, foot_x, dt=0.02, foot_y=foot_y)["step_width"] == pytest.approx(0.22)


def test_alternation_is_how_often_the_leading_foot_switches() -> None:
    contact, foot_x = walking()  # the leading foot switches every double-support phase
    assert gait_summary(contact, foot_x, dt=0.02)["alternation"] == pytest.approx(1.0, abs=0.1)
    same_foot_leads = np.tile([0.12, -0.12], (len(contact), 1))  # left always in front
    assert gait_summary(contact, same_foot_leads, dt=0.02)["alternation"] == 0.0
