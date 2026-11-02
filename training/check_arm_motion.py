"""Check that the arms really move with the plan in the simulator: actual shoulder pitch and elbow ranges.

    source ~/isaacsim-env/bin/activate && cd ~/g1_voice_ws/training && python check_arm_motion.py
"""

import os
import sys

import warp as wp

wp.config.enable_backward = False
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walk_task  # noqa: E402,F401
from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args  # noqa: E402
from isaaclab_rl.entrypoints.common import close_env, create_isaaclab_env  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

TASK = "R1-Walk-Plan-Flat-Play"
args_cli = _parse_args(["--task", TASK, "--num_envs", "4", "--viz", "none", "physics=isaacsim_physx"])
env_cfg, _ = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True)
env_cfg.scene.num_envs = 4
env_cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
env_cfg.commands.base_velocity.rel_standing_envs = 0.0
env_cfg.commands.base_velocity.debug_vis = False
with launch_simulation(env_cfg, args_cli):
    import math

    import torch

    env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
    base = env.unwrapped
    robot = base.scene["robot"]
    names = ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_elbow_joint", "right_elbow_joint"]
    ids = robot.find_joints(names, preserve_order=True)[0]
    command = base.command_manager.get_term("base_velocity")
    env.reset()
    targets, actual = [], []
    for step in range(int(3.0 / base.step_dt)):
        command.vel_command_b[:] = torch.tensor([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.0, 0.8]],
                                                device=base.device)
        env.step(torch.zeros(4, base.action_manager.total_action_dim, device=base.device))
        if step > 1.0 / base.step_dt:
            targets.append(base.action_manager.get_term("joint_pos").arm_targets.clone())
            actual.append(robot.data.joint_pos.torch[:, ids].clone())
    targets, actual = torch.stack(targets), torch.stack(actual)
    for robot_index, label in enumerate(["forward 0.5", "forward 1.0", "backward 0.4", "turn 0.8"]):
        line = []
        for joint, name in enumerate(["L shoulder", "R shoulder", "L elbow", "R elbow"]):
            t, a = targets[:, robot_index, joint], actual[:, robot_index, joint]
            line.append(f"{name}: target {math.degrees(t.max() - t.min()):4.0f} deg, actual {math.degrees(a.max() - a.min()):4.0f} deg")
        print(f"{label:13s} " + " | ".join(line), flush=True)
    close_env(env)
