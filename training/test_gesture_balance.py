"""Test balance during gestures, headless: robots play the voice assistant's gestures standing and walking.

Each robot plays one gesture from the gesture library (the real keyframes, at their real speed), set the way the
walk scene sets them, standing or walking at 0.4 m/s. Prints, per gesture, how many robots fell.

    source ~/isaacsim-env/bin/activate && cd ~/g1_voice_ws/training
    python test_gesture_balance.py --checkpoint PATH
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

import warp as wp

wp.config.enable_backward = False

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, os.path.join(TRAINING_DIR, "..", "src", "robot_voice_assistant"))
import walk_task  # noqa: E402,F401  (registers the task)
from robot_voice_assistant.gestures import GESTURE_KEYFRAMES, NEUTRAL_POSE, GestureTrajectory  # noqa: E402
from walk_scene import configure_for_deployment  # noqa: E402
from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args  # noqa: E402
from isaaclab_rl.entrypoints.common import close_env, create_isaaclab_env  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner  # noqa: E402
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

GESTURES = sorted(g for g in GESTURE_KEYFRAMES if g != "none")
SPEEDS = (0.0, 0.4)  # standing, walking forward (m/s)
ROBOTS_EACH = 2  # per gesture and speed
FALLEN_BELOW_M = 0.5
START_S, SECONDS = 1.0, 7.0  # each gesture starts after 1 s; the test runs this long


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--robots", type=int, default=ROBOTS_EACH, help="robots per gesture and speed")
    parser.add_argument("--gestures", nargs="*", default=GESTURES)
    parser.add_argument("--arm-speed", type=float, default=None, help="override the walk's arm speed limit (rad/s)")
    own_args, launcher_args = parser.parse_known_args()
    gestures, robots = own_args.gestures, own_args.robots
    cases = [(gesture, speed) for speed in SPEEDS for gesture in gestures for _ in range(robots)]
    count = len(cases)
    args_cli = _parse_args(["--task", "R1-Walk-Plan-Flat-Play", "--num_envs", str(count), "--viz", "none",
                            *launcher_args, "physics=isaacsim_physx"])
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True)
    configure_for_deployment(env_cfg, "none", follow_camera=False)  # exactly as the voice robot's scene
    env_cfg.scene.num_envs = count
    env_cfg.terminations.base_contact = None  # a fallen robot stays down, so it is counted
    if own_args.arm_speed is not None:
        env_cfg.actions.joint_pos.arm_speed = own_args.arm_speed

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        import torch

        from walk_task.plan_action import ARM_SWING_JOINTS

        agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli), check_rsl_rl_version())
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(own_args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        print(f"[test_gesture_balance] Policy {own_args.checkpoint}", flush=True)

        base_env = env.unwrapped
        device, dt = base_env.device, base_env.step_dt
        robot = base_env.scene["robot"]
        walk_term = base_env.action_manager.get_term("joint_pos")
        names = list(robot.joint_names)
        other = [j for j in NEUTRAL_POSE if j in names and j not in ARM_SWING_JOINTS]
        other_ids = [names.index(j) for j in other]
        everyone = torch.arange(count, device=device)
        command = torch.tensor([[speed, 0.0, 0.0] for _, speed in cases], device=device)
        trajectories = {g: GestureTrajectory(g, dict(NEUTRAL_POSE)) for g in gestures}
        lowest = torch.full((count,), 9.0, device=device)
        obs = env.get_observations()
        for tick in range(int(SECONDS / dt)):
            elapsed = tick * dt - START_S
            poses = {g: trajectories[g].sample(elapsed) if elapsed > 0 else dict(NEUTRAL_POSE) for g in gestures}
            arms = torch.tensor([[poses[g][j] for j in ARM_SWING_JOINTS] for g, _ in cases], device=device)
            rest = torch.tensor([[poses[g][j] for j in other] for g, _ in cases], device=device)
            base_env.command_manager.get_term("base_velocity").vel_command_b[:] = command
            walk_term.set_arm_pose(arms)
            walk_term.set_gesture_targets(other_ids, rest)
            robot.actuators.target_command.set_position_index(value=rest, joint_ids=other_ids, env_ids=everyone)
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
            lowest = torch.minimum(lowest, robot.data.root_pos_w.torch[:, 2])

        fell = (lowest < FALLEN_BELOW_M).tolist()
        print(f"\n{'gesture':14s} standing  walking", flush=True)
        for gesture in gestures:
            row = [sum(f for f, (g, s) in zip(fell, cases) if g == gesture and s == speed) for speed in SPEEDS]
            print(f"{gesture:14s} {row[0]}/{robots} fell   {row[1]}/{robots} fell", flush=True)
        print(f"\nRESULT: {sum(fell)}/{count} robots fell", flush=True)


if __name__ == "__main__":
    main()
