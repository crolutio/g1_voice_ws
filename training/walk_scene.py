"""Isaac Sim scene with a walking Unitree R1, driven by the trained policy and by ROS 2.

The robot is free (not pinned). The trained policy moves the legs; /cmd_vel sets the walking speed and
turning rate, and /robot/joint_command (from gesture_node) moves the arms and waist, as in the standing scene.

    ~/g1_voice_ws/start_isaac_walk.sh                    # latest trained policy
    ~/g1_voice_ws/start_isaac_walk.sh --checkpoint PATH  # a specific model_*.pt
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time

import warp as wp

wp.config.enable_backward = False

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAINING_DIR)
import walk_task  # noqa: E402,F401  (registers the task)
from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args  # noqa: E402
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import check_rsl_rl_version  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config  # noqa: E402

TASK = "R1-Walk-Flat-Play"


def configure_for_deployment(env_cfg, viewer: str, follow_camera: bool) -> None:
    """One robot, commands from ROS instead of random ones, no time limit, and the chosen viewer."""

    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1.0e6
    env_cfg.events.animate_upper_body = None  # the arms and waist follow /robot/joint_command instead
    env_cfg.events.reset_base = None
    command = env_cfg.commands.base_velocity
    command.resampling_time_range = (1.0e9, 1.0e9)
    command.rel_standing_envs = 0.0
    command.debug_vis = False
    if viewer == "kit" and follow_camera:
        from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg

        env_cfg.sim.visualizer_cfgs = [
            KitVisualizerCfg(origin_type="asset", origin_track_path="robot", eye=(2.4, -1.2, 0.9), lookat=(0.0, 0.0, 0.1))
        ]
    elif viewer == "newton_gl":
        from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonGLVisualizerCfg

        env_cfg.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(window_width=1280, window_height=720, eye=(3.0, -3.0, 1.6), lookat=(0.0, 0.0, 0.7))
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="", help="model_*.pt to load (default: latest training run)")
    parser.add_argument("--headless", action="store_true", help="run without any window (same as --viewer none)")
    parser.add_argument(
        "--viewer",
        choices=["newton_gl", "kit", "none"],
        default="newton_gl",
        help="newton_gl: light OpenGL window (fast); kit: full Isaac Sim window (slow on some laptops)",
    )
    parser.add_argument("--no-follow", action="store_true", help="do not move the camera with the robot")
    own_args, launcher_args = parser.parse_known_args()

    viewer = "none" if own_args.headless else own_args.viewer
    argv = ["--task", TASK, "--num_envs", "1", "--viz", viewer, *launcher_args]
    args_cli = _parse_args([*argv, "physics=isaacsim_physx"])
    check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True)
    configure_for_deployment(env_cfg, viewer, follow_camera=not own_args.no_follow)

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        import torch

        from walk_ros_bridge import WalkRosBridge

        agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli), check_rsl_rl_version())
        apply_env_overrides(args_cli, env_cfg)
        log_root = os.path.join(TRAINING_DIR, "logs", "rsl_rl", agent_cfg.experiment_name)
        checkpoint = own_args.checkpoint or get_checkpoint_path(log_root, ".*", r"model_.*\.pt")
        print(f"[walk_scene] Loading policy {checkpoint}", flush=True)

        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        command_term = base_env.command_manager.get_term("base_velocity")
        joint_names = list(robot.joint_names)
        joint_index = {name: i for i, name in enumerate(joint_names)}
        env_ids = torch.tensor([0], device=base_env.device)
        bridge = WalkRosBridge()
        cleanup.callback(bridge.shutdown)

        step_dt = base_env.step_dt
        sim_time = 0.0
        obs = env.get_observations()
        report_wall, report_sim = time.monotonic(), 0.0
        print("[walk_scene] Walking. Listening on /cmd_vel and /robot/joint_command.", flush=True)
        while True:
            started = time.monotonic()
            bridge.spin_once()
            command_term.vel_command_b[0] = torch.tensor(bridge.velocity_command(), device=base_env.device)
            targets = {name: value for name, value in bridge.upper_body_targets().items() if name in joint_index}
            if targets:
                robot.actuators.target_command.set_position_index(
                    value=torch.tensor([list(targets.values())], device=base_env.device),
                    joint_ids=[joint_index[name] for name in targets],
                    env_ids=env_ids,
                )
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
            sim_time += step_dt

            data = robot.data
            bridge.publish(
                sim_time,
                joint_names,
                data.joint_pos.torch[0].tolist(),
                data.root_pos_w.torch[0].tolist(),
                data.root_quat_w.torch[0].tolist(),
                data.root_lin_vel_b.torch[0].tolist(),
                data.root_ang_vel_b.torch[0].tolist(),
            )
            if time.monotonic() - report_wall >= 5.0:
                speed = (sim_time - report_sim) / (time.monotonic() - report_wall)
                print(f"[walk_scene] simulation speed {speed:.0%} of real time", flush=True)
                report_wall, report_sim = time.monotonic(), sim_time
            # Run no faster than real time, so speeds and gestures look natural.
            time.sleep(max(0.0, step_dt - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
