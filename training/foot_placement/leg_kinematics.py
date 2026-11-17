"""Unitree R1 leg kinematics in torch: foot pose from joint angles, and IK to place a foot.

Each leg is the URDF chain pelvis -> hip pitch -> hip roll -> hip yaw -> knee -> ankle pitch -> ankle roll.
The "foot" is the ankle roll link's frame (at the ankle; its x axis points forward along the sole). Everything
is batched, so it runs for thousands of simulated robots at once on the GPU.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass

import torch

LEG_JOINTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
# The bent-knee standing pose of the walking task: IK starts from here (a straight knee is a poor start)
STANDING = {"hip_pitch": -0.25, "knee": 0.5, "ankle_pitch": -0.25}
# Safe mode (walking): joint angles stay in a normal walking range (inside the URDF limits), so an unreachable
# target can never flip the leg into a strange pose; each solve moves a joint at most MAX_JOINT_STEP_RAD; and a
# target beyond the leg's reach is pulled in to it (else the solver can fold the leg instead of stretching it).
# Roll and yaw ranges are symmetric, so they suit both legs.
WALKING_RANGE = {"hip_pitch": (-1.4, 0.7), "hip_roll": (-0.5, 0.5), "hip_yaw": (-0.6, 0.6), "knee": (-0.2, 2.0),
                 "ankle_pitch": (-0.9, 0.6), "ankle_roll": (-0.3, 0.3)}
MAX_JOINT_STEP_RAD = 0.15  # per solve (one 20 ms control step): 7.5 rad/s
CONVERGED_ERROR = 1e-5  # m and rad: IK on the CPU stops iterating once the foot is this close to its target
REACH_FRACTION = 1.0  # targets beyond the farthest reachable point are pulled in to it


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    cr, sr = torch.cos(torch.tensor(roll)), torch.sin(torch.tensor(roll))
    cp, sp = torch.cos(torch.tensor(pitch)), torch.sin(torch.tensor(pitch))
    cy, sy = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
    rz = torch.tensor([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = torch.tensor([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (N, 3, 3) about a unit ``axis`` (3,) by ``angle`` (N,)."""
    x, y, z = axis
    c, s = torch.cos(angle), torch.sin(angle)
    t = 1 - c
    return torch.stack(
        [
            torch.stack([t * x * x + c, t * x * y - s * z, t * x * z + s * y], -1),
            torch.stack([t * x * y + s * z, t * y * y + c, t * y * z - s * x], -1),
            torch.stack([t * x * z - s * y, t * y * z + s * x, t * z * z + c], -1),
        ],
        -2,
    )


def rotation_error(rotation: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Rotation vector (N, 3) that turns ``rotation`` into ``target`` (both (N, 3, 3)), in the parent frame."""
    error = target @ rotation.transpose(-1, -2)
    skew = torch.stack(
        [error[:, 2, 1] - error[:, 1, 2], error[:, 0, 2] - error[:, 2, 0], error[:, 1, 0] - error[:, 0, 1]], -1
    )
    cos_angle = ((error.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)
    sin_angle = torch.sin(angle)
    scale = torch.where(sin_angle > 1e-6, angle / (2 * sin_angle), torch.full_like(angle, 0.5))
    return skew * scale.unsqueeze(-1)


@dataclass
class _Joint:
    origin_position: torch.Tensor  # (3,) in the parent link frame
    origin_rotation: torch.Tensor  # (3, 3)
    axis: torch.Tensor  # (3,) unit, in the joint frame


class LegKinematics:
    def __init__(self, side: str, joints: list[_Joint], lower: torch.Tensor, upper: torch.Tensor) -> None:
        self.side = side
        self.joint_names = [f"{side}_{name}_joint" for name in LEG_JOINTS]
        self._joints = joints
        self.lower_limits, self.upper_limits = lower, upper
        walking = torch.tensor([WALKING_RANGE[name] for name in LEG_JOINTS])
        self.walking_lower = torch.maximum(lower, walking[:, 0])
        self.walking_upper = torch.minimum(upper, walking[:, 1])
        # the leg's reach: the farthest the foot gets from the hip anywhere in the walking range
        generator = torch.Generator().manual_seed(0)
        samples = self.walking_lower + (self.walking_upper - self.walking_lower) * torch.rand(4096, 6, generator=generator)
        samples[:2048, 3] = self.walking_lower[3]  # half of them with the knee straight
        stretched, _ = self.foot_pose(samples)
        self.reach = REACH_FRACTION * float(torch.linalg.norm(stretched - self.hip_position(), dim=1).max())

    @classmethod
    def from_urdf(cls, path: str, side: str) -> LegKinematics:
        root = ElementTree.parse(path).getroot()
        by_name = {joint.get("name"): joint for joint in root.findall("joint")}
        joints, lower, upper = [], [], []
        for name in LEG_JOINTS:
            element = by_name[f"{side}_{name}_joint"]
            origin = element.find("origin")
            xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
            axis = torch.tensor([float(v) for v in element.find("axis").get("xyz").split()])
            joints.append(_Joint(torch.tensor(xyz), _rpy_matrix(*rpy), axis / torch.linalg.norm(axis)))
            limit = element.find("limit")
            lower.append(float(limit.get("lower")))
            upper.append(float(limit.get("upper")))
        return cls(side, joints, torch.tensor(lower), torch.tensor(upper))

    def to(self, device: torch.device | str) -> LegKinematics:
        for joint in self._joints:
            joint.origin_position = joint.origin_position.to(device)
            joint.origin_rotation = joint.origin_rotation.to(device)
            joint.axis = joint.axis.to(device)
        self.lower_limits, self.upper_limits = self.lower_limits.to(device), self.upper_limits.to(device)
        self.walking_lower, self.walking_upper = self.walking_lower.to(device), self.walking_upper.to(device)
        return self

    def hip_position(self) -> torch.Tensor:
        """(1, 3): where the leg starts (the hip pitch joint) in the pelvis frame."""
        return self._joints[0].origin_position.unsqueeze(0)

    def default_angles(self) -> torch.Tensor:
        return torch.tensor([[STANDING.get(name, 0.0) for name in LEG_JOINTS]], device=self.lower_limits.device)

    def _forward(self, angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Foot position (N, 3), rotation (N, 3, 3), and each joint's axis and position (N, 6, 3) in the pelvis."""
        count = angles.shape[0]
        position = torch.zeros(count, 3, device=angles.device)
        rotation = torch.eye(3, device=angles.device).expand(count, 3, 3)
        axes, points = [], []
        for index, joint in enumerate(self._joints):
            position = position + rotation @ joint.origin_position
            rotation = rotation @ joint.origin_rotation
            axes.append(rotation @ joint.axis)
            points.append(position)
            rotation = rotation @ axis_angle_matrix(joint.axis, angles[:, index])
        return position, rotation, torch.stack(axes, 1), torch.stack(points, 1)

    def foot_pose(self, angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Foot (ankle roll link) position (N, 3) and rotation (N, 3, 3) in the pelvis frame."""
        position, rotation, _, _ = self._forward(angles)
        return position, rotation

    def solve(
        self,
        target_position: torch.Tensor,
        target_rotation: torch.Tensor,
        start: torch.Tensor,
        iterations: int = 10,
        damping: float = 0.03,
        safe: bool = False,
    ) -> torch.Tensor:
        """Joint angles (N, 6) that put the foot at the target pose (damped least squares, within the limits).

        Warm-start from the previous solution: one walking step changes little, so a few iterations suffice.
        ``safe`` (for walking): the target is pulled in to the leg's reach, the joints stay in WALKING_RANGE and
        move at most MAX_JOINT_STEP_RAD, so a far target gives the closest sensible leg, never a flipped one.
        """
        start = start.expand(target_position.shape[0], 6)
        angles = start.clone()
        lower, upper = (self.walking_lower, self.walking_upper) if safe else (self.lower_limits, self.upper_limits)
        if safe:
            hip = self.hip_position().to(target_position.device)
            offset = target_position - hip
            distance = torch.linalg.norm(offset, dim=1, keepdim=True).clamp(min=1e-6)
            target_position = hip + offset * (distance.clamp(max=self.reach) / distance)  # pulled in to the reach
            angles = torch.clamp(angles, lower, upper)
        identity = torch.eye(6, device=angles.device)
        # On the CPU (one robot, live scene) stop as soon as every foot is on target: warm-started from the last
        # step, that is usually after 1-2 iterations. On the GPU (training) the check would stall the GPU every
        # iteration, so the fixed number of iterations is kept there.
        early_exit = angles.device.type == "cpu"
        for _ in range(iterations):
            position, rotation, axes, points = self._forward(angles)
            error = torch.cat([target_position - position, rotation_error(rotation, target_rotation)], dim=1)
            if early_exit and float(error.abs().max()) < CONVERGED_ERROR:
                break
            linear = torch.cross(axes, position.unsqueeze(1) - points, dim=-1)  # (N, 6, 3)
            jacobian = torch.cat([linear, axes], dim=-1).transpose(1, 2)  # (N, 6 task, 6 joints)
            step = jacobian.transpose(1, 2) @ torch.linalg.solve(
                jacobian @ jacobian.transpose(1, 2) + damping**2 * identity, error.unsqueeze(-1)
            )
            angles = torch.clamp(angles + step.squeeze(-1), lower, upper)
        if safe:
            angles = start + (angles - start).clamp(-MAX_JOINT_STEP_RAD, MAX_JOINT_STEP_RAD)
        return angles
