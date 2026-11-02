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


# --- step rhythm: walk foot by foot instead of hopping on both feet (the knee policy jumped 26-36% of the time) ---


def both_feet_in_air(in_contact: torch.Tensor) -> torch.Tensor:
    """Per robot: 1 while no foot touches the ground (a jump), else 0."""
    return (in_contact.sum(dim=1) == 0).float()


def desired_stance(phase: torch.Tensor, stance_fraction: float) -> torch.Tensor:
    """Which feet should be on the ground at ``phase`` (0-1) of the step cycle: (robots, [left, right]).

    The left foot is down for the first ``stance_fraction`` of the cycle, the right foot for the same share
    half a cycle later, so both are down briefly twice per cycle and never both in the air.
    """
    left = phase < stance_fraction
    right = torch.remainder(phase + 0.5, 1.0) < stance_fraction
    return torch.stack([left, right], dim=1)


def gait_phase_match(
    in_contact: torch.Tensor, phase: torch.Tensor, moving: torch.Tensor, stance_fraction: float
) -> torch.Tensor:
    """Per robot: share of feet (0, 0.5 or 1) doing what the step clock asks; standing robots keep both down."""
    wanted = torch.where(moving.unsqueeze(1), desired_stance(phase, stance_fraction), torch.ones_like(in_contact))
    return (in_contact == wanted).float().mean(dim=1)


def stride_length(
    in_contact: torch.Tensor, feet_x: torch.Tensor, walking: torch.Tensor, max_length: float
) -> torch.Tensor:
    """Per robot: how far one foot is ahead of the other (m, capped) while both are down and walking."""
    both_down = in_contact.all(dim=1)
    length = torch.clamp(torch.abs(feet_x[:, 0] - feet_x[:, 1]), max=max_length)
    return length * (both_down & walking)


# --- the human gait: the hip swings the leg forward with a bent knee, the knee straightens before landing,
# the foot lands in front (behind when walking backward), heel first, and the straight leg takes the weight ---


def swing_knee_shortfall(
    knee_pos: torch.Tensor, in_contact: torch.Tensor, moving: torch.Tensor, target: float
) -> torch.Tensor:
    """Per robot: how far (rad) the knees of swinging feet are bent less than ``target``, while moving."""
    shortfall = torch.clamp(target - knee_pos, min=0.0) * (~in_contact)
    return shortfall.sum(dim=1) * moving


def landing_knee_excess(knee_pos: torch.Tensor, first_contact: torch.Tensor, max_bend: float) -> torch.Tensor:
    """Per robot: how far (rad) a foot that just touched down has its knee bent more than ``max_bend``."""
    return (torch.clamp(knee_pos - max_bend, min=0.0) * first_contact).sum(dim=1)


def landing_in_front(
    feet_x: torch.Tensor, first_contact: torch.Tensor, direction: torch.Tensor, max_distance: float
) -> torch.Tensor:
    """Per robot: how far (m, capped) a landing foot is ahead of the hips in the walking direction.

    ``feet_x``: feet relative to the hips along the heading; ``direction``: +1 forward, -1 backward, 0 neither.
    """
    ahead = torch.clamp(feet_x * direction.unsqueeze(1), min=0.0, max=max_distance)
    return (ahead * first_contact).sum(dim=1)


def heel_first(toe_up: torch.Tensor, first_contact: torch.Tensor, max_tilt: float) -> torch.Tensor:
    """Per robot: toe-up tilt (sine of the angle, capped) of feet that just touched down."""
    return (torch.clamp(toe_up, min=0.0, max=max_tilt) * first_contact).sum(dim=1)


def stance_knee_excess(knee_pos: torch.Tensor, in_contact: torch.Tensor, max_bend: float) -> torch.Tensor:
    """Per robot: how far (rad) the knees of legs on the ground are bent more than ``max_bend`` (lunging).

    The step-rhythm policy walked in deep lunges, both feet down 80-89% of the time and knees bent up to 124 deg.
    """
    return (torch.clamp(knee_pos - max_bend, min=0.0) * in_contact).sum(dim=1)


# --- closer legs (the step-rhythm policy walked with its feet 37-53 cm apart; the hips are about 22 cm apart),
# and each foot steps past the other (left in front, then right in front, ...) ---


def step_width_error(feet_y: torch.Tensor, min_width: float, max_width: float) -> torch.Tensor:
    """Per robot: how far (m) the side-to-side distance between the feet is outside [min_width, max_width]."""
    width = torch.abs(feet_y[:, 0] - feet_y[:, 1])
    return torch.clamp(width - max_width, min=0.0) + torch.clamp(min_width - width, min=0.0)


def landing_passes_other_foot(
    feet_x: torch.Tensor, first_contact: torch.Tensor, direction: torch.Tensor, max_distance: float
) -> torch.Tensor:
    """Per robot: how far (m, capped) a landing foot is past the other foot in the walking direction.

    Walking forward, each foot must land ahead of the other one, so the feet take turns in front; walking
    backward (``direction`` -1), behind it.
    """
    ahead_of_other = torch.stack([feet_x[:, 0] - feet_x[:, 1], feet_x[:, 1] - feet_x[:, 0]], dim=1)
    passed = torch.clamp(ahead_of_other * direction.unsqueeze(1), min=0.0, max=max_distance)
    return (passed * first_contact).sum(dim=1)


# --- quaternions: Isaac Lab 3.0 stores them as (x, y, z, w) ---


def heading_from_quat_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Heading (yaw, rad) of (..., 4) quaternions in (x, y, z, w) order."""
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def toe_up_from_quat_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Upward part (sine of the toe-up angle) of a foot frame's forward (x) axis; (x, y, z, w) quaternions."""
    x, y, z, w = quat.unbind(-1)
    return 2 * (x * z - w * y)


def sideways_capture_point_outside(
    sideways_velocity: torch.Tensor, height: torch.Tensor, feet_y: torch.Tensor, margin: float, gravity: float = 9.81
) -> torch.Tensor:
    """How far (m) the capture point is sideways outside the feet (0 inside); a measure of falling sideways.

    The capture point is where the robot would have to put a foot to stop its motion: the pelvis position plus
    velocity / omega, with omega = sqrt(g / height) (linear inverted pendulum). ``sideways_velocity`` (N,) and
    ``feet_y`` (N, 2) are in the heading frame, relative to the pelvis; ``height`` (N,) is the pelvis height.
    """
    omega = torch.sqrt(gravity / height.clamp(min=0.2))
    capture_y = sideways_velocity / omega
    left_edge = feet_y.max(dim=1).values + margin
    right_edge = feet_y.min(dim=1).values - margin
    return (capture_y - left_edge).clamp(min=0) + (right_edge - capture_y).clamp(min=0)
