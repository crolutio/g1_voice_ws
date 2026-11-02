"""Show the planned walk (foot planter + IK, no balance control) on the R1 in Isaac Sim.

The robot is posed directly every frame: the hips move along the planned line, upright, and the legs take the
IK joint angles, so this shows exactly the plan, not what the robot could balance by itself. Sequence: stand
straight, walk forward, stop, walk backward, stop, turn left, stop.

    source ~/isaacsim-env/bin/activate && cd ~/g1_voice_ws/training && python play_foot_plan.py
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import struct
import sys
import time

import warp as wp

wp.config.enable_backward = False

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAINING_DIR)
import walk_task  # noqa: E402,F401  (registers the task)
from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args  # noqa: E402
from isaaclab_rl.entrypoints.common import close_env, create_isaaclab_env  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

from foot_placement.gait_planner import FootPlanter, GaitPlan, cycle_period  # noqa: E402
from foot_placement.leg_kinematics import LegKinematics  # noqa: E402

TASK = "R1-Walk-Flat-Play"
URDF = os.path.expanduser("~/g1_voice_ws/robots/r1/R1.urdf")
SEQUENCE = [  # (label, command forward m/s, left m/s, turn rad/s, seconds)
    ("standing straight", 0.0, 0.0, 0.0, 2.0), ("walking forward 1.0 m/s", 1.0, 0.0, 0.0, 5.0),
    ("stop", 0.0, 0.0, 0.0, 1.5), ("walking backward 0.5 m/s", -0.5, 0.0, 0.0, 4.0), ("stop", 0.0, 0.0, 0.0, 1.5),
    ("turning left 1.0 rad/s", 0.0, 0.0, 1.0, 4.0),
    ("stop", 0.0, 0.0, 0.0, 2.0),
]
WALK_IN_S = 0.8


def sole_depth(stl_path: str) -> float:
    """How far (m) the lowest point of a binary STL mesh is below its link origin."""
    import numpy as np

    with open(stl_path, "rb") as file:
        data = file.read()
    count = struct.unpack("<I", data[80:84])[0]
    triangles = np.frombuffer(data[84 : 84 + count * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return float(-triangles["v"].reshape(-1, 3)[:, 2].min())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--loops", type=int, default=3, help="how many times to play the sequence")
    own_args, launcher_args = parser.parse_known_args()
    args_cli = _parse_args(["--task", TASK, "--num_envs", "1", "--viz", "newton_gl", *launcher_args,
                            "physics=isaacsim_physx"])
    env_cfg, _ = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True)
    env_cfg.scene.num_envs = 1
    env_cfg.events.animate_upper_body = None
    env_cfg.commands.base_velocity.debug_vis = False

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        import torch

        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        base_env = env.unwrapped
        device = base_env.device
        robot = base_env.scene["robot"]
        dt = base_env.step_dt
        legs = {side: LegKinematics.from_urdf(URDF, side).to(device) for side in ("left", "right")}
        leg_ids = {side: robot.find_joints(legs[side].joint_names, preserve_order=True)[0] for side in legs}
        joints = robot.data.default_joint_pos.torch[0:1].clone()  # upper body: relaxed default pose
        origin = base_env.scene.env_origins[0, :2]
        plan = GaitPlan()
        angles = {side: legs[side].default_angles() for side in legs}

        # how high the ankle (foot link origin) sits above the ground with the sole flat: the lowest point of the
        # foot mesh, below the ankle (a fixed property of the robot; measuring a standing robot was unreliable)
        ankle_height = sole_depth(os.path.join(os.path.dirname(URDF), "meshes", "left_ankle_roll_link.STL"))
        print(f"[play_foot_plan] ankle height above the ground: {ankle_height * 100:.1f} cm", flush=True)

        for loop in range(own_args.loops):
            planter = FootPlanter(plan, count=1, device=device)
            hips_xy, hips_yaw = torch.zeros(1, 2, device=device), torch.zeros(1, device=device)
            for label, vx, vy, wz, seconds in SEQUENCE:
                print(f"[play_foot_plan] {label}", flush=True)
                command = torch.tensor([[vx, vy, wz]], device=device)
                period = float(cycle_period(plan, command)[0])
                cycle = plan.stance_fraction  # the left foot takes the first step
                started = 0.0
                for _ in range(int(seconds / dt)):
                    frame_start = time.monotonic()
                    amount = 0.5 - 0.5 * math.cos(math.pi * min(started / WALK_IN_S, 1.0))
                    phase = torch.tensor([cycle % 1.0], device=device)
                    cycle += dt / period
                    feet = planter.step(hips_xy, hips_yaw, command, phase, torch.tensor([amount], device=device))
                    for index, side in enumerate(("left", "right")):
                        angles[side] = legs[side].solve(feet.position[:, index], feet.rotation[:, index], angles[side])
                        joints[:, leg_ids[side]] = angles[side]
                    walk_height = plan.stand_foot_z_m + (plan.foot_z_m - plan.stand_foot_z_m) * amount
                    hips_z = ankle_height - walk_height
                    # root pose: position, then the quaternion in Isaac Lab's (x, y, z, w) order (upright, yawed)
                    yaw = float(hips_yaw[0])
                    pose = torch.tensor([[float(origin[0] + hips_xy[0, 0]), float(origin[1] + hips_xy[0, 1]), hips_z,
                                          0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]], device=device)
                    robot.write_root_pose_to_sim_index(root_pose=pose)
                    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=device))
                    robot.write_joint_position_to_sim_index(position=joints)
                    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joints))
                    robot.set_joint_position_target_index(target=joints)
                    base_env.scene.write_data_to_sim()
                    base_env.sim.step(render=True)
                    base_env.scene.update(dt=base_env.physics_dt)
                    c, s = math.cos(yaw), math.sin(yaw)
                    velocity = torch.tensor([[c * vx - s * vy, s * vx + c * vy]], device=device)
                    hips_xy = hips_xy + dt * amount * velocity
                    hips_yaw = hips_yaw + dt * amount * wz
                    started += dt
                    time.sleep(max(0.0, dt - (time.monotonic() - frame_start)))
        print("[play_foot_plan] done", flush=True)


if __name__ == "__main__":
    main()
