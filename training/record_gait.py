"""Record how the policy walks (body positions and foot contacts) for animate_gait.py.

    source ~/isaacsim-env/bin/activate && cd ~/g1_voice_ws/training
    python record_gait.py --checkpoint PATH --out gait.npz
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

import numpy as np
import warp as wp

wp.config.enable_backward = False

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAINING_DIR)
import walk_task  # noqa: E402,F401  (registers the task)
from walk_task.reward_math import heading_from_quat_xyzw  # noqa: E402
from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args  # noqa: E402
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner  # noqa: E402
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config  # noqa: E402

TASK = "R1-Walk-Flat-Play"
COMMANDS = {"forward 0.5 m/s": (0.5, 0.0, 0.0), "backward 0.3 m/s": (-0.3, 0.0, 0.0), "turn left 0.8 rad/s": (0.0, 0.0, 0.8)}
BODIES = [
    "pelvis_link", "waist_yaw_link", "head_yaw_link",
    "left_hip_pitch_link", "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "left_shoulder_pitch_link", "left_elbow_link", "left_wrist_roll_link",
    "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_roll_link",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--out", default=os.path.join(TRAINING_DIR, "gait_recording.npz"))
    own_args, launcher_args = parser.parse_known_args()
    num_envs = len(COMMANDS)
    args_cli = _parse_args(["--task", TASK, "--num_envs", str(num_envs), "--viz", "none", *launcher_args,
                            "physics=isaacsim_physx"])
    check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True)
    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = 1.0e6
    env_cfg.events.animate_upper_body = None
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
        print(f"[record_gait] Policy {checkpoint}", flush=True)
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
        body_ids, _ = robot.find_bodies(BODIES, preserve_order=True)
        command_term = base_env.command_manager.get_term("base_velocity")
        targets = torch.tensor(list(COMMANDS.values()), device=base_env.device, dtype=torch.float32)

        warmup, measured = int(2.0 / base_env.step_dt), int(own_args.seconds / base_env.step_dt)
        positions, contacts, yaw = [], [], []
        obs = env.get_observations()
        for step in range(warmup + measured):
            command_term.vel_command_b[:] = targets
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
            if step < warmup:
                continue
            data = robot.data
            positions.append((data.body_pos_w.torch[:, body_ids] - base_env.scene.env_origins[:, None]).cpu().numpy())
            contacts.append((sensor.data.current_contact_time.torch[:, foot_ids] > 0).cpu().numpy())
            yaw.append(heading_from_quat_xyzw(data.root_quat_w.torch).cpu().numpy())
        np.savez_compressed(
            own_args.out, positions=np.stack(positions, 1), contacts=np.stack(contacts, 1), yaw=np.stack(yaw, 1),
            commands=np.array(list(COMMANDS)), bodies=np.array(BODIES), dt=base_env.step_dt, checkpoint=checkpoint,
        )
        print(f"[record_gait] saved {own_args.out}", flush=True)


if __name__ == "__main__":
    main()
