"""Gait measurements from foot contacts and foot positions (plain numpy, testable without Isaac Sim)."""

from __future__ import annotations

import numpy as np


def gait_summary(
    contact: np.ndarray, foot_x: np.ndarray, dt: float, foot_y: np.ndarray | None = None
) -> dict[str, float]:
    """Describe one robot's gait.

    ``contact``: (steps, 2) bool, left and right foot on the ground. ``foot_x``: (steps, 2) foot positions
    along the robot's heading, in metres. Returns the share of time with both feet in the air ("flight",
    i.e. jumping or running), with one foot down, with both down; the mean front-to-back distance between
    the feet while both are down ("step_length": one foot forward of the other); steps per second; and,
    with ``foot_y`` (sideways positions), the mean side-to-side distance between the feet ("step_width");
    and "alternation": how often the foot in front switches between double-support phases (1 = every time).
    """
    feet_down = contact.sum(axis=1)
    double = feet_down == 2
    step_length = float(np.mean(np.abs(foot_x[double, 0] - foot_x[double, 1]))) if double.any() else 0.0
    touchdowns = int(np.sum(contact[1:] & ~contact[:-1]))
    # which foot is in front in each double-support phase, and how often that switches (1 = left, right, left, ...)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], double.astype(int), [0]])))
    leaders = [np.sign(foot_x[(a + b - 1) // 2, 0] - foot_x[(a + b - 1) // 2, 1]) for a, b in zip(edges[::2], edges[1::2])]
    switches = [x != y for x, y in zip(leaders, leaders[1:])]
    return {
        "flight": float(np.mean(feet_down == 0)),
        "single_stance": float(np.mean(feet_down == 1)),
        "double_support": float(np.mean(double)),
        "step_length": step_length,
        "steps_per_s": touchdowns / (len(contact) * dt),
        "alternation": float(np.mean(switches)) if switches else 0.0,
        "step_width": float(np.mean(np.abs(foot_y[:, 0] - foot_y[:, 1]))) if foot_y is not None else 0.0,
    }
