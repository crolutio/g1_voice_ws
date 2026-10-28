"""Reward maths for walking with bent knees (plain torch, so it can be tested without Isaac Sim).

The first R1 policy learned to lock its knees straight on their -10 degree stop and shuffle with the hips
and ankles: the stop carries the weight for free and nothing asked for foot lift. These two terms make that
gait cost something.
"""

from __future__ import annotations

import torch


def knee_straightness(knee_pos: torch.Tensor, min_bend: float) -> torch.Tensor:
    """Per robot: how far (rad, summed over knees) the knees are straighter than ``min_bend``."""
    return torch.sum(torch.clamp(min_bend - knee_pos, min=0.0), dim=1)


def swing_foot_height_shortfall(foot_z: torch.Tensor, in_contact: torch.Tensor, target: float) -> torch.Tensor:
    """Per robot: squared shortfall of the swinging foot below ``target`` metres above the standing foot.

    Only judged while exactly one foot is on the ground (the other is swinging); measuring against the
    standing foot makes it independent of where the foot link's origin sits.
    """
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    stance_z = torch.where(in_contact, foot_z, torch.full_like(foot_z, float("inf"))).min(dim=1).values
    lift = foot_z - stance_z.unsqueeze(1)
    shortfall = torch.clamp(target - lift, min=0.0) ** 2
    shortfall = torch.where(in_contact, torch.zeros_like(shortfall), shortfall)
    return torch.where(single_stance, shortfall.sum(dim=1), torch.zeros_like(stance_z))


def drift_while_standing(base_velocity: torch.Tensor, command: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """Per robot: squared body speed (vx, vy, and half the turning rate) while told to stand still.

    The knee-bending policy kept creeping forward at about 0.25 m/s for seconds after a stop.
    """
    standing = (torch.linalg.norm(command[:, :2], dim=1) < threshold) & (torch.abs(command[:, 2]) < threshold)
    drift = base_velocity[:, 0] ** 2 + base_velocity[:, 1] ** 2 + 0.5 * base_velocity[:, 2] ** 2
    return drift * standing
