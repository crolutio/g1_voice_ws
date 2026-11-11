"""A visitor who walks up to the robot's stall holding something out: the arm pose, the hand, and the visit."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stall_visit import StallVisit, hand_track, hold_out_pose  # noqa: E402


def standing_body() -> np.ndarray:
    """A stick figure facing +x: a torso column (|y| <= 0.15) and a left arm hanging at y = 0.25."""
    torso = np.array([(0.0, y, z) for y in (-0.15, 0.0, 0.15) for z in np.arange(0.0, 1.8, 0.05)])
    arm = np.array([(0.0, 0.25, z) for z in np.arange(0.70, 1.42, 0.02)])
    return np.concatenate([torso, arm]).astype(np.float32)


def test_holding_out_raises_the_arm_forward_and_leaves_the_body_alone() -> None:
    body = standing_body()
    held, hand_ids, hand = hold_out_pose(body)
    torso = np.abs(body[:, 1]) <= 0.15
    assert np.allclose(held[torso], body[torso])  # only the arm moves
    assert len(hand_ids) > 0 and np.all(body[hand_ids, 2] < 0.86)  # the hand: the bottom of the arm
    assert hand[0] > 0.25  # held out in front of the body
    assert 0.8 < hand[2] < 1.1  # at about waist height, level with the robot's camera
    upper = (body[:, 1] > 0.2) & (body[:, 2] > 1.15) & (body[:, 2] < 1.25)  # the upper arm keeps its length
    shoulder = np.array([0.0, 0.25, 1.40])
    assert np.allclose(np.linalg.norm(held[upper] - shoulder, axis=1), np.linalg.norm(body[upper] - shoulder, axis=1),
                       atol=0.03)


def test_the_hand_is_followed_through_the_walk() -> None:
    body = standing_body()
    _, hand_ids, _ = hold_out_pose(body)
    frames = np.stack([body + np.array([0.1 * k, 0.0, 0.0], np.float32) for k in range(3)])
    track = hand_track(frames, hand_ids)
    assert track.shape == (3, 3)
    assert np.allclose(np.diff(track[:, 0]), 0.1, atol=1e-5)


def visit() -> StallVisit:
    return StallVisit(entry=(4.6, -5.0), stop=(1.35, 0.0), exit=(4.6, 5.0), robot=(0.0, 0.0), speed=1.0,
                      cycle_m=2.8, frames=80, standing_frame=40, max_stay_s=30.0)


def test_a_visitor_walks_up_stops_facing_the_robot_and_leaves_when_told() -> None:
    v = visit()
    assert v.update(0.0).state == "idle"
    assert v.request("apple", 1.0)
    walking = v.update(2.0)
    assert walking.state == "approaching" and walking.frame is not None and walking.held == "apple"
    there = v.update(1.0 + v.approach_s + 0.1)
    assert there.state == "at_stall" and there.frame is None  # standing, holding the object out
    assert (there.x, there.y) == (1.35, 0.0)
    assert math.isclose(math.cos(there.yaw), -1.0, abs_tol=1e-6)  # facing the robot (-x)
    assert not v.request("mug", 20.0)  # one visitor at a time
    v.leave(20.0)
    assert v.update(21.0).state == "leaving"
    assert v.update(20.0 + v.leave_s + 0.1).state == "idle"
    assert v.request("mug", 40.0)


def test_a_visitor_nobody_sends_away_leaves_on_their_own() -> None:
    v = visit()
    v.request("banana", 0.0)
    assert v.update(v.approach_s + 29.0).state == "at_stall"
    assert v.update(v.approach_s + 31.0).state == "leaving"


def test_the_feet_keep_pace_with_the_ground_on_the_way_in() -> None:
    v = visit()
    v.request("bottle", 0.0)
    a, b = v.update(0.5), v.update(0.5 + 2.8 / 80)  # one frame's worth of ground at 1 m/s
    assert (b.frame - a.frame) % 80 == 1
