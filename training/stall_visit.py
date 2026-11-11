"""A visitor who walks up to the robot's stall holding something, stops in front of it, and leaves when told.

Used to test the robot's vision (stall_demo.py): the visitor carries the object in the left hand while walking and,
at the stall, holds it out towards the robot's head camera. The arm is raised by turning the walk character's left
arm about its shoulder and elbow (the character has no holding pose of its own).

Character frame: metres, z up, facing +x, feet on z = 0; its left arm hangs at y > 0.19 to 0.21.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# the left arm: points further out than this (by height: the hips reach 0.20 m, the chest 0.18 m) below the shoulder
ARM_SIDE_Y = ((0.90, 1.10), (0.21, 0.19))
SHOULDER_Z = 1.42
SHOULDER_BLEND_M = 0.17  # the turn fades in over this much arm below the shoulder, so the skin does not tear
HAND_Z = 0.86  # the arm below this height is the hand


def _turn_xz(points: np.ndarray, pivot: np.ndarray, degrees: np.ndarray | float) -> np.ndarray:
    """Turn points about the y axis through ``pivot``; positive raises a hanging arm forward (towards +x)."""
    angle = np.radians(degrees)
    dx, dz = points[:, 0] - pivot[0], points[:, 2] - pivot[2]
    out = points.copy()
    out[:, 0] = pivot[0] + dx * np.cos(angle) - dz * np.sin(angle)
    out[:, 2] = pivot[2] + dx * np.sin(angle) + dz * np.cos(angle)
    return out


def hold_out_pose(standing: np.ndarray, raise_deg: float = 10.0, elbow_deg: float = 40.0):
    """The standing character holding something out in its left hand: (points, hand point ids, hand centre)."""
    arm = (standing[:, 1] > np.interp(standing[:, 2], *ARM_SIDE_Y)) & (standing[:, 2] < SHOULDER_Z)
    ids = np.flatnonzero(arm)
    pts = standing[ids].astype(np.float64)
    shoulder = np.array([pts[:, 0].mean(), 0.0, SHOULDER_Z])
    elbow_z = SHOULDER_Z - 0.30
    # the elbow bends first (forearm only), then the whole arm turns at the shoulder
    forearm = pts[:, 2] < elbow_z
    elbow = np.array([pts[:, 0].mean(), 0.0, elbow_z])
    pts[forearm] = _turn_xz(pts[forearm], elbow, elbow_deg)
    blend = np.clip((SHOULDER_Z - standing[ids, 2]) / SHOULDER_BLEND_M, 0.0, 1.0)
    pts = _turn_xz(pts, shoulder, raise_deg * blend)
    held = standing.copy()
    held[ids] = pts.astype(standing.dtype)
    hand_ids = ids[standing[ids, 2] < HAND_Z]
    return held, hand_ids, held[hand_ids].mean(axis=0)


def hand_track(frames: np.ndarray, hand_ids: np.ndarray) -> np.ndarray:
    """The hand's centre in every walk frame: (frames, 3)."""
    return frames[:, hand_ids].mean(axis=1)


@dataclass(frozen=True)
class VisitPose:
    state: str  # "idle", "approaching", "at_stall" or "leaving"
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    frame: int | None = None  # walk frame; None while standing at the stall holding the object out
    held: str | None = None  # what the visitor carries


class StallVisit:
    """One visitor at a time: ``request(object)`` sends them from ``entry`` to ``stop`` (facing ``robot``), where
    they stay until ``leave()`` or ``max_stay_s``; then they walk to ``exit`` and are gone."""

    def __init__(self, entry, stop, exit, robot, speed: float, cycle_m: float, frames: int, standing_frame: int,
                 max_stay_s: float = 40.0) -> None:
        self.entry, self.stop, self.exit, self.robot = entry, stop, exit, robot
        self.speed, self.cycle_m, self.frames, self.standing_frame = speed, cycle_m, frames, standing_frame
        self.max_stay_s = max_stay_s
        self.approach_s = math.dist(entry, stop) / speed
        self.leave_s = math.dist(stop, exit) / speed
        self._held: str | None = None
        self._start = self._leave_at = None

    def request(self, held: str, now: float) -> bool:
        """Send a visitor carrying ``held``; False if one is already on the way or there."""
        if self.update(now).state != "idle":
            return False
        self._held, self._start, self._leave_at = held, now, None
        return True

    def leave(self, now: float) -> None:
        state = self.update(now).state
        if state == "at_stall":
            self._leave_at = now
        elif state == "approaching":  # turn round as soon as they arrive
            self._leave_at = self._start + self.approach_s

    def update(self, now: float) -> VisitPose:
        if self._start is None:
            return VisitPose("idle")
        t = now - self._start
        if t < self.approach_s:
            return self._walking("approaching", self.entry, self.stop, t)
        leave_at = self._leave_at if self._leave_at is not None else self._start + self.approach_s + self.max_stay_s
        if now < leave_at:
            yaw = math.atan2(self.robot[1] - self.stop[1], self.robot[0] - self.stop[0])
            return VisitPose("at_stall", self.stop[0], self.stop[1], yaw, None, self._held)
        t = now - leave_at
        if t < self.leave_s:
            return self._walking("leaving", self.stop, self.exit, t)
        self._start = None
        return VisitPose("idle")

    def _walking(self, state: str, a, b, t: float) -> VisitPose:
        length = math.dist(a, b)
        share = min(1.0, t * self.speed / length) if length > 0 else 1.0
        x, y = a[0] + share * (b[0] - a[0]), a[1] + share * (b[1] - a[1])
        frame = int(round(t * self.speed / self.cycle_m * self.frames)) % self.frames
        return VisitPose(state, x, y, math.atan2(b[1] - a[1], b[0] - a[0]), frame, self._held)
