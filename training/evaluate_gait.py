"""Measure how the trained policy walks: jumping (both feet in the air), steps, step length, knees.

Runs several robots per command (forward, backward, turning, standing) headless and prints a table.

    source ~/isaacsim-env/bin/activate && cd ~/g1_voice_ws/training
    python evaluate_gait.py                       # newest trained policy
    python evaluate_gait.py --checkpoint PATH     # a specific model_*.pt
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys

import numpy as np
import warp as wp

wp.config.enable_backward = False

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAINING_DIR)
import walk_task  # noqa: E402,F401  (registers the task)
from walk_task.reward_math import heading_from_quat_xyzw  # noqa: E402
from gait_metrics import gait_summary  # noqa: E402
from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args  # noqa: E402
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner  # noqa: E402
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config  # noqa: E402

TASK = "R1-Walk-Flat-Play"
COMMANDS = {  # name: (forward m/s, sideways m/s, turn rad/s)
    "forward 0.5 m/s": (0.5, 0.0, 0.0),
    "forward 1.0 m/s": (1.0, 0.0, 0.0),
    "backward 0.3 m/s": (-0.3, 0.0, 0.0),
    "turn left 0.8 rad/s": (0.0, 0.0, 0.8),
    "turn right 0.8 rad/s": (0.0, 0.0, -0.8),
    "stand still": (0.0, 0.0, 0.0),
}
ROBOTS_PER_COMMAND = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--seconds", type=float, default=10.0, help="measured time per robot (after 2 s)")
    own_args, launcher_args = parser.parse_known_args()
    num_envs = ROBOTS_PER_COMMAND * len(COMMANDS)
    args_cli = _parse_args(["--task", TASK, "--num_envs", str(num_envs), "--viz", "none", *launcher_args,
                            "physics=isaacsim_physx"])
    check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True)
    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = 1.0e6
    env_cfg.events.animate_upper_body = None  # plain walking, arms at rest
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    command_cfg.rel_standing_envs = 0.0
    command_cfg.debug_vis = False

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        import torch

        agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli), check_rsl_rl_version())
        apply_env_overrides(args_cli, env_cfg)
        log_root = os.path.join(TRAINING_DIR, "logs", "rsl_rl", agent_cfg.experiment_name)
        checkpoint = own_args.checkpoint or get_checkpoint_path(log_root, ".*", r"model_.*\.pt")
        print(f"[evaluate_gait] Policy {checkpoint}", flush=True)
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        sensor = base_env.scene.sensors["contact_forces"]
        foot_ids, _ = sensor.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)
        robot_foot_ids, _ = robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)
        knee_ids, _ = robot.find_joints(["left_knee_joint", "right_knee_joint"], preserve_order=True)
        command_term = base_env.command_manager.get_term("base_velocity")
        targets = torch.tensor([c for c in COMMANDS.values() for _ in range(ROBOTS_PER_COMMAND)],
                               device=base_env.device, dtype=torch.float32)

        warmup, measured = int(2.0 / base_env.step_dt), int(own_args.seconds / base_env.step_dt)
        contacts, feet_x, feet_y, knees, heights, velocity = [], [], [], [], [], []
        fell = torch.zeros(num_envs, dtype=torch.bool)
        obs = env.get_observations()
        for step in range(warmup + measured):
            command_term.vel_command_b[:] = targets
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
            fell |= dones.bool().cpu()
            if step < warmup:
                continue
            data = robot.data
            yaw = heading_from_quat_xyzw(data.root_quat_w.torch)  # Isaac Lab 3.0 quaternions are (x, y, z, w)
            offset = data.body_pos_w.torch[:, robot_foot_ids, :2] - data.root_pos_w.torch[:, None, :2]
            along_heading = offset[..., 0] * torch.cos(yaw)[:, None] + offset[..., 1] * torch.sin(yaw)[:, None]
            sideways = -offset[..., 0] * torch.sin(yaw)[:, None] + offset[..., 1] * torch.cos(yaw)[:, None]
            contacts.append((sensor.data.current_contact_time.torch[:, foot_ids] > 0).cpu().numpy())
            feet_x.append(along_heading.cpu().numpy())
            feet_y.append(sideways.cpu().numpy())
            knees.append(data.joint_pos.torch[:, knee_ids].cpu().numpy())
            heights.append(data.root_pos_w.torch[:, 2].cpu().numpy())
            velocity.append(torch.cat([data.root_lin_vel_b.torch[:, :1], data.root_ang_vel_b.torch[:, 2:3]], 1).cpu().numpy())

        contacts, feet_x, feet_y, knees = np.stack(contacts, 1), np.stack(feet_x, 1), np.stack(feet_y, 1), np.stack(knees, 1)
        heights, velocity = np.stack(heights, 1), np.stack(velocity, 1)
        print(f"\n{'command':22s} {'jumping':>8s} {'one foot':>9s} {'both feet':>10s} {'step len':>9s} "
              f"{'steps/s':>8s} {'alternate':>9s} {'width':>7s} {'knee range':>13s} {'bounce':>7s} {'speed':>13s} falls")
        for index, name in enumerate(COMMANDS):
            ids = range(index * ROBOTS_PER_COMMAND, (index + 1) * ROBOTS_PER_COMMAND)
            stats = [gait_summary(contacts[i], feet_x[i], base_env.step_dt, feet_y[i]) for i in ids]
            mean = {key: float(np.mean([s[key] for s in stats])) for key in stats[0]}
            knee = np.degrees(knees[list(ids)])
            bounce = float(np.mean([np.std(heights[i]) for i in ids]))
            vx, wz = velocity[list(ids)].mean(axis=(0, 1))
            print(f"{name:22s} {mean['flight']:8.0%} {mean['single_stance']:9.0%} {mean['double_support']:10.0%} "
                  f"{mean['step_length'] * 100:7.1f}cm {mean['steps_per_s']:8.2f} {mean['alternation']:9.0%} {mean['step_width'] * 100:5.1f}cm "
                  f"{knee.min():5.0f} to {knee.max():3.0f}d {bounce * 100:5.1f}cm "
                  f"{vx:+.2f}m/s {wz:+.2f}r/s {int(fell[list(ids)].sum())}")
        print("jumping = both feet in the air; step len = how far one foot is ahead of the other while both are down;"
              " alternate = how often the foot in front switches (100% = left, right, left, ...); width = feet apart sideways;"
              " bounce = up-and-down motion of the pelvis (std)", flush=True)
        _ = math  # imported for interactive use


if __name__ == "__main__":
    main()
