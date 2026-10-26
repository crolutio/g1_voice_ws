"""Forward kinematics and arm clearance checks for the Unitree R1 upper body.

The model (joint frames from Unitree's URDF and point clouds sampled from its meshes) is stored in
``config/robot_collision_model.npz``; ``scripts/build_collision_model.py`` regenerates it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "robot_collision_model.npz")

TORSO_LINK = "waist_yaw_link"  # the arms and the head hang off this link
HEAD_LINK = "head_yaw_link"
# Which parts of each arm are checked against which body part. The forearm is checked against the
# head only: next to the torso it is close by design (arms hanging relaxed).
CLEARANCE_CHECKS: dict[tuple[str, str], tuple[str, ...]] = {
    ("left", HEAD_LINK): ("left_wrist_roll_link", "left_elbow_link"),
    ("right", HEAD_LINK): ("right_wrist_roll_link", "right_elbow_link"),
    ("left", TORSO_LINK): ("left_wrist_roll_link",),
    ("right", TORSO_LINK): ("right_wrist_roll_link",),
}
BODY_LINKS = (HEAD_LINK, TORSO_LINK)
ARM_LINKS = tuple(sorted({link for links in CLEARANCE_CHECKS.values() for link in links}))


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4 transform from parent link frame to joint frame
    axis: np.ndarray  # rotation axis in the joint frame (zero for fixed joints)


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis / np.linalg.norm(axis)
    c, s, t = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def origin_transform(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    rotation = (
        rotation_about_axis(np.array([0.0, 0.0, 1.0]), yaw)
        @ rotation_about_axis(np.array([0.0, 1.0, 0.0]), pitch)
        @ rotation_about_axis(np.array([1.0, 0.0, 0.0]), roll)
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = xyz
    return transform


class RobotKinematics:
    def __init__(self, joints: Sequence[Joint], link_points: Mapping[str, np.ndarray]) -> None:
        self._joints_by_child = {joint.child: joint for joint in joints}
        self.link_points = dict(link_points)

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> RobotKinematics:
        data = np.load(path, allow_pickle=False)
        table = json.loads(str(data["joints"]))
        joints = [
            Joint(j["name"], j["parent"], j["child"], np.array(j["origin"]), np.array(j["axis"])) for j in table
        ]
        points = {key[len("points/"):]: data[key] for key in data.files if key.startswith("points/")}
        return cls(joints, points)

    def link_transform(self, link: str, joint_angles: Mapping[str, float], root: str = TORSO_LINK) -> np.ndarray:
        """4x4 pose of ``link`` in the frame of ``root`` (an ancestor link)."""
        transform = np.eye(4)
        current = link
        while current != root:
            joint = self._joints_by_child.get(current)
            if joint is None:
                raise KeyError(f"{root} is not an ancestor of {link}")
            motion = np.eye(4)
            if np.any(joint.axis):
                motion[:3, :3] = rotation_about_axis(joint.axis, joint_angles.get(joint.name, 0.0))
            transform = joint.origin @ motion @ transform
            current = joint.parent
        return transform

    def points_in(self, link: str, joint_angles: Mapping[str, float], root: str = TORSO_LINK) -> np.ndarray:
        transform = self.link_transform(link, joint_angles, root)
        points = self.link_points[link]
        return points @ transform[:3, :3].T + transform[:3, 3]

    def hand_clearance(self, joint_angles: Mapping[str, float], side: str, body_link: str) -> float:
        """Smallest distance in metres between an arm (see CLEARANCE_CHECKS) and a body link."""
        return self.clearances(joint_angles)[(side, body_link)]

    def clearances(self, joint_angles: Mapping[str, float]) -> dict[tuple[str, str], float]:
        """Arm-to-body distances for both arms and every body link, keyed by (side, body_link)."""
        cache: dict[str, np.ndarray] = {}

        def pose_of(link: str) -> np.ndarray:
            if link == TORSO_LINK:
                return np.eye(4)
            if link not in cache:
                joint = self._joints_by_child[link]
                motion = np.eye(4)
                if np.any(joint.axis):
                    motion[:3, :3] = rotation_about_axis(joint.axis, joint_angles.get(joint.name, 0.0))
                cache[link] = pose_of(joint.parent) @ joint.origin @ motion
            return cache[link]

        def points(link: str) -> np.ndarray:
            transform = pose_of(link)
            return self.link_points[link] @ transform[:3, :3].T + transform[:3, 3]

        result = {}
        bodies = {link: points(link) for link in BODY_LINKS}
        arms = {link: points(link) for link in ARM_LINKS}
        for (side, body_link), links in CLEARANCE_CHECKS.items():
            arm = np.concatenate([arms[link] for link in links])
            body = bodies[body_link]
            # |a - b|^2 = |a|^2 + |b|^2 - 2 a.b, computed for all pairs with one matrix product.
            squared = (arm**2).sum(axis=1)[:, None] + (body**2).sum(axis=1)[None, :] - 2.0 * arm @ body.T
            result[(side, body_link)] = float(np.sqrt(max(float(squared.min()), 0.0)))
        return result
