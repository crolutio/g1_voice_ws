"""The people around the robot: as its range sensor would report them, and as solid bodies in the physics.

In the hall the visitors are animated meshes, which the physics does not know about. Each one also gets an
invisible kinematic capsule that follows them, so the robot cannot walk through a person: it would bump into them,
as in real life. The robot is kept from getting that close in the first place by /robot/nearby_people (the
people in its own frame, as a lidar or depth camera around a real robot reports them), which the motion and
gesture nodes use to keep their distance (robot_voice_assistant/safety.py).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

SENSOR_RANGE_M = 6.0
PERSON_RADIUS_M = 0.22  # shoulders about 0.45 m wide
PERSON_HEIGHT_M = 1.74
PARKED_XY = (500.0, 500.0)  # where the bodies of people not in the hall wait, far from everything

World = tuple[float, float] | None  # a person's position on the floor, None when they are not there


def robot_frame(people: Sequence[World], robot_xy: tuple[float, float], robot_yaw: float,
                max_range_m: float = SENSOR_RANGE_M) -> list[tuple[float, float]]:
    """Each person within range, as (forward, left) metres from the robot."""
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    seen = []
    for person in people:
        if person is None:
            continue
        dx, dy = person[0] - robot_xy[0], person[1] - robot_xy[1]
        if math.hypot(dx, dy) <= max_range_m:
            seen.append((c * dx + s * dy, -s * dx + c * dy))
    return seen


def body_positions(people: Sequence[World], height_m: float = PERSON_HEIGHT_M) -> list[tuple[float, float, float]]:
    """Where each person's capsule goes (its middle at half their height); people not there wait far away."""
    return [(person[0], person[1], height_m / 2) if person is not None
            else (PARKED_XY[0] + 3.0 * n, PARKED_XY[1], height_m / 2) for n, person in enumerate(people)]


def configure_people_bodies(env_cfg, count: int) -> None:
    """Add ``count`` invisible kinematic capsules to the scene (before it is created)."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    for n, position in enumerate(body_positions([None] * count)):
        setattr(env_cfg.scene, f"person_body_{n}", RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/PersonBody{n}",
            spawn=sim_utils.CapsuleCfg(
                radius=PERSON_RADIUS_M, height=PERSON_HEIGHT_M - 2 * PERSON_RADIUS_M, axis="Z", visible=False,
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),  # moved by us, not by forces
                collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=70.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        ))


class PeopleBodies:
    """Moves the capsules to where the people are (call every physics step)."""

    def __init__(self, scene, count: int) -> None:
        import torch

        self._torch = torch
        self._bodies = [scene[f"person_body_{n}"] for n in range(count)]

    def update(self, people: Sequence[World]) -> None:
        for body, (x, y, z) in zip(self._bodies, body_positions(people)):
            pose = self._torch.tensor([[x, y, z, 0.0, 0.0, 0.0, 1.0]], device=body.device)  # (x, y, z, qx, qy, qz, qw)
            body.write_root_pose_to_sim_index(root_pose=pose)
