"""Test the walking policy the way the voice assistant uses it: falls, balance and speed, with a pass/fail result.

Each scenario is a list of (command, seconds) steps; several robots run each scenario at once (headless), each
starting with the random push training uses. A robot has fallen when its pelvis drops below 0.5 m. Prints per
scenario the torso tilt, speed and turn errors (once each command has settled) and the falls, then PASS/FAIL.

    source ~/isaacsim-env/bin/activate && cd ~/g1_voice_ws/training
    python test_falls.py                       # newest plan-walk policy
    python test_falls.py --checkpoint PATH     # a specific model_*.pt
"""

from __future__ import annotations

import argparse
import math
import contextlib
import os
import sys

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

TASK = "R1-Walk-Plan-Flat-Play"
STAND, FORWARD, BACKWARD = (0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (-0.3, 0.0, 0.0)
VOICE_SEQUENCE = [(STAND, 2.0), (FORWARD, 6.0), (STAND, 2.0), (BACKWARD, 5.0), (STAND, 2.0), ((0.0, 0.0, 0.5), 6.4),
                  ((0.0, 0.0, -0.5), 6.4), (FORWARD, 5.0), (STAND, 3.0)]  # what drive_walk_scene.py sends
SCENARIOS = {  # name: [(forward m/s, sideways m/s, turn rad/s), seconds), ...]
    "forward": [(STAND, 2.0), (FORWARD, 8.0)],
    "backward from standing": [(STAND, 2.0), (BACKWARD, 8.0)],
    "fast backward from standing": [(STAND, 2.0), ((-0.5, 0.0, 0.0), 8.0)],
    "forward, stop, backward": [(STAND, 2.0), (FORWARD, 4.0), (STAND, 2.0), (BACKWARD, 6.0)],
    "forward straight to backward": [(STAND, 2.0), (FORWARD, 4.0), (BACKWARD, 6.0)],
    "turn left, then right": [(STAND, 2.0), ((0.0, 0.0, 0.8), 4.0), ((0.0, 0.0, -0.8), 4.0)],
    "walk, stop, walk": [(STAND, 2.0), (FORWARD, 3.0), (STAND, 2.0), (FORWARD, 3.0), (STAND, 2.0)],
    "backward, stop, turn left": [(STAND, 2.0), (BACKWARD, 5.0), (STAND, 2.0), ((0.0, 0.0, 0.8), 4.0)],
    "forward, stop, turn right": [(STAND, 2.0), (FORWARD, 5.0), (STAND, 2.0), ((0.0, 0.0, -0.8), 4.0)],
    "voice robot sequence": VOICE_SEQUENCE,
    "voice robot sequence x3": VOICE_SEQUENCE[1:] * 3,
    "turned 80 deg first": [((0.0, 0.0, -0.8), 2.0), (STAND, 2.0)] + VOICE_SEQUENCE,
    "walked 5 m first": [(FORWARD, 16.0), (STAND, 2.0)] + VOICE_SEQUENCE,
    "stood 70 s first": [(STAND, 70.0)] + VOICE_SEQUENCE,
    "voice robot sequence x2": VOICE_SEQUENCE[1:] * 2,
}
FALLEN_BELOW_M = 0.5
WAYS = ("front", "back", "left", "right")
MAX_TILT_DEG = 10.0  # the torso stays upright: never more than this from vertical
MAX_SPEED_ERROR = 0.15  # m/s, average once a command has settled (forward plus sideways error)
MAX_TURN_ERROR = 0.2  # rad/s, average once a command has settled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default="", nargs="*", help="one or more model_*.pt, tested one after another")
    parser.add_argument("--robots", type=int, default=16, help="robots per scenario")
    parser.add_argument("--deployment", action="store_true", help="the walk scene's settings (planner on the CPU, ...)")
    parser.add_argument("--gestures", action="store_true", help="random gestures as in training (much harder)")
    parser.add_argument("--turn-speed", type=float, default=None, help="rad/s for the voice sequence's turns")
    parser.add_argument("--trace", action="store_true", help="print 4 robots of the first scenario at 74-80 s")
    parser.add_argument("--only", nargs="*", default=None, help="run only these scenarios")
    parser.add_argument("--spacing", type=float, default=None, help="metres between the robots (default 2.5)")
    parser.add_argument("--no-com", action="store_true", help="turn the centre-of-mass correction off")
    parser.add_argument("--debug-scenario", default="", help="print the plan and hip twist of this scenario's robots")
    parser.add_argument("--times", action="store_true", help="also print when each robot fell")
    parser.add_argument("--no-policy", action="store_true", help="follow the plan alone (policy corrections off)")
    own_args, launcher_args = parser.parse_known_args()
    if own_args.turn_speed:
        for name, steps in SCENARIOS.items():
            SCENARIOS[name] = [((c[0], c[1], math.copysign(own_args.turn_speed, c[2]) if c[2] else 0.0), t * (0.8 / own_args.turn_speed if c[2] else 1.0))
                               for c, t in steps]
    if own_args.only:
        for name in list(SCENARIOS):
            if name not in own_args.only:
                del SCENARIOS[name]
    names = list(SCENARIOS)
    num_envs = own_args.robots * len(names)
    args_cli = _parse_args(["--task", TASK, "--num_envs", str(num_envs), "--viz", "none", *launcher_args,
                            "physics=isaacsim_physx"])
    check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True)
    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = 1.0e6
    if own_args.deployment:
        from walk_scene import configure_for_deployment

        configure_for_deployment(env_cfg, "none", follow_camera=False)
        env_cfg.scene.num_envs = num_envs
        env_cfg.episode_length_s = 1.0e6
    if own_args.spacing:
        env_cfg.scene.env_spacing = own_args.spacing
    if own_args.no_com:
        env_cfg.actions.joint_pos.com_compensation = False
    if not own_args.gestures:  # the voice robot gestures through the gesture node, tested in test_gesture_balance.py
        env_cfg.events.animate_upper_body = None
    for name in ("base_contact", "lost_balance", "hips_too_low"):  # a fallen robot stays down, so we can count it
        setattr(env_cfg.terminations, name, None)
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    command_cfg.rel_standing_envs = 0.0
    command_cfg.debug_vis = False

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        import torch

        agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli), check_rsl_rl_version())
        apply_env_overrides(args_cli, env_cfg)
        log_root = os.path.join(TRAINING_DIR, "logs", "rsl_rl", agent_cfg.experiment_name)
        checkpoints = own_args.checkpoint or [get_checkpoint_path(log_root, ".*", r"model_.*\.pt")]
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        for checkpoint in checkpoints:
            print(f"\n[test_falls] Policy {checkpoint}", flush=True)
            runner.load(checkpoint)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            with torch.inference_mode():  # the env's tensors were made in inference mode by the previous test
                env.reset()
            test_policy(env, policy, own_args, num_envs)


def test_policy(env, policy, own_args, num_envs: int) -> None:
    import torch

    names = list(SCENARIOS)

    base_env = env.unwrapped
    device, dt = base_env.device, base_env.step_dt
    robot = base_env.scene["robot"]
    command_term = base_env.command_manager.get_term("base_velocity")
    scenario_of = torch.arange(num_envs, device=device) // own_args.robots
    fell_in_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    fell_at = torch.full((num_envs,), -1.0, device=device)
    fell_toward = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    tilt_max, tilt_sum = torch.zeros(num_envs, device=device), torch.zeros(num_envs, device=device)
    speed_error_sum, turn_error_sum = torch.zeros(num_envs, device=device), torch.zeros(num_envs, device=device)
    settled_ticks = torch.zeros(num_envs, device=device)
    longest = max(sum(seconds for _, seconds in steps) for steps in SCENARIOS.values())
    obs = env.get_observations()
    for tick in range(int(longest / dt)):
        now = tick * dt
        commands = torch.zeros(num_envs, 3, device=device)
        step_now = torch.zeros(num_envs, dtype=torch.long, device=device)
        into_step = torch.zeros(num_envs, device=device)  # seconds since this robot's command changed
        for index, name in enumerate(names):
            elapsed, chosen, which = 0.0, STAND, len(SCENARIOS[name]) - 1
            for number, (command, seconds) in enumerate(SCENARIOS[name]):
                if now < elapsed + seconds:
                    chosen, which, started = command, number, elapsed
                    break
                elapsed += seconds
            else:
                chosen, started = STAND, elapsed
            rows = scenario_of == index
            into_step[rows] = now - started
            commands[rows] = torch.tensor(chosen, device=device)
            step_now[rows] = which
        command_term.vel_command_b[:] = commands
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(torch.zeros_like(actions) if own_args.no_policy else actions)
        down = (robot.data.root_pos_w.torch[:, 2] < FALLEN_BELOW_M) & (fell_in_step < 0)
        fell_in_step[down] = step_now[down]
        fell_at[down] = now
        # which way it tipped: the pelvis velocity at the moment it went down, in its own heading frame
        yaw = heading_from_quat_xyzw(robot.data.root_quat_w.torch)
        velocity = robot.data.root_lin_vel_w.torch[:, :2]
        ahead = velocity[:, 0] * torch.cos(yaw) + velocity[:, 1] * torch.sin(yaw)
        left = -velocity[:, 0] * torch.sin(yaw) + velocity[:, 1] * torch.cos(yaw)
        direction = torch.where(ahead.abs() > left.abs(), torch.where(ahead > 0, 0, 1), torch.where(left > 0, 2, 3))
        fell_toward[down] = direction[down]
        if own_args.trace and 74.0 <= now <= 80.0 and tick % 5 == 0:
            first = int((scenario_of == 0).nonzero()[0])
            term = base_env.action_manager.get_term("joint_pos")
            for robot_id in range(first, first + 4):
                feet_yaw = torch.atan2(term.feet.rotation[robot_id, :, 1, 0], term.feet.rotation[robot_id, :, 0, 0])
                hip = robot.data.joint_pos.torch[robot_id, [term._joint_ids[2], term._joint_ids[8]]]
                plan_hip = term.reference[robot_id, [2, 8]]
                print(f"TRACE r{robot_id - first} t={now:5.2f} z {float(robot.data.root_pos_w.torch[robot_id, 2]):4.2f}"
                      f" phase {float(term.phase[robot_id]):4.2f} stance {term.feet.stance[robot_id].int().tolist()}"
                      f" plan foot yaw {feet_yaw[0]:5.2f} {feet_yaw[1]:5.2f} ref hip yaw {plan_hip[0]:5.2f} {plan_hip[1]:5.2f}"
                      f" hip yaw {hip[0]:5.2f} {hip[1]:5.2f} wz {float(robot.data.root_ang_vel_b.torch[robot_id, 2]):5.2f}"
                      f" yaw {float(torch.rad2deg(yaw[robot_id])):7.1f}", flush=True)

        if own_args.debug_scenario and tick % int(0.5 / dt) == 0:
            rows = (scenario_of == names.index(own_args.debug_scenario)) & (fell_in_step < 0)
            if rows.any():
                term = base_env.action_manager.get_term("joint_pos")
                hip_yaw = robot.data.joint_pos.torch[:, [term._joint_ids[2], term._joint_ids[8]]]
                feet_yaw = torch.atan2(term.feet.rotation[:, :, 1, 0], term.feet.rotation[:, :, 0, 0])
                print(f"DEBUG t={now:5.1f} up {int(rows.sum()):2d}  hip yaw max {float(hip_yaw[rows].abs().max()):4.2f}"
                      f"  plan foot yaw max {float(feet_yaw[rows].abs().max()):4.2f}"
                      f"  walk_in {float(term._walk_in[rows.to(term._walk_in.device)].mean()):4.2f}"
                      f"  com shift {float(term._com_shift()[rows].abs().max()):5.3f}"
                      f"  yaw {float(torch.rad2deg(yaw[rows]).mean()):7.1f}"
                      f"  x {float(robot.data.root_pos_w.torch[rows, 0].mean() - base_env.scene.env_origins[rows, 0].mean()):5.2f}",
                      flush=True)

        # balance and speed, for robots still up, once each command has settled (1 s after it changes)
        up = fell_in_step < 0
        settled = up & (into_step > 1.0)
        gravity = robot.data.projected_gravity_b.torch
        tilt = torch.rad2deg(torch.atan2(torch.linalg.norm(gravity[:, :2], dim=1), -gravity[:, 2]))
        tilt_max = torch.where(settled, torch.maximum(tilt_max, tilt), tilt_max)
        tilt_sum += torch.where(settled, tilt, torch.zeros_like(tilt))
        body = robot.data.root_lin_vel_b.torch
        speed_error = torch.abs(body[:, 0] - commands[:, 0]) + torch.abs(body[:, 1] - commands[:, 1])
        turn_error = torch.abs(robot.data.root_ang_vel_b.torch[:, 2] - commands[:, 2])
        speed_error_sum += torch.where(settled, speed_error, torch.zeros_like(speed_error))
        turn_error_sum += torch.where(settled, turn_error, torch.zeros_like(turn_error))
        settled_ticks += settled.float()

    ticks = settled_ticks.clamp(min=1)
    print(f"\n{'scenario':32s} tilt avg / max   speed error   turn error")
    for index, name in enumerate(names):
        rows = (scenario_of == index) & (fell_in_step < 0)
        if rows.any():
            print(f"{name:32s} {float((tilt_sum / ticks)[rows].mean()):4.1f} / {float(tilt_max[rows].max()):4.1f} deg"
                  f"   {float((speed_error_sum / ticks)[rows].mean()):4.2f} m/s"
                  f"     {float((turn_error_sum / ticks)[rows].mean()):4.2f} rad/s", flush=True)
    print(f"\n{'scenario':32s} fell  (in which step)")
    for index, name in enumerate(names):
        steps = fell_in_step[scenario_of == index].tolist()
        fallen = [s for s in steps if s >= 0]
        where = ", ".join(f"{fallen.count(s)} while {describe(SCENARIOS[name][s][0])}" for s in sorted(set(fallen)))
        times = sorted(t for t in fell_at[scenario_of == index].tolist() if t >= 0)
        ways = [WAYS[w] for w in fell_toward[scenario_of == index].tolist() if w >= 0]
        toward = f"  toward: {', '.join(f'{ways.count(w)} {w}' for w in WAYS if w in ways)}" if ways else ""
        times = sorted(t for t in fell_at[scenario_of == index].tolist() if t >= 0)
        when = f"  at {', '.join(f'{t:.1f}' for t in times)} s" if own_args.times and times else ""
        print(f"{name:32s} {len(fallen):2d}/{len(steps)}  {where}{toward}{when}", flush=True)
    everyone = [WAYS[w] for w in fell_toward.tolist() if w >= 0]
    print(f"\nall falls: {len(everyone)}/{num_envs}; toward " + ", ".join(f"{w} {everyone.count(w)}" for w in WAYS), flush=True)
    up = fell_in_step < 0
    worst_tilt = float(tilt_max[up].max()) if up.any() else 90.0
    speed = float((speed_error_sum / ticks)[up].mean()) if up.any() else 9.0
    turn = float((turn_error_sum / ticks)[up].mean()) if up.any() else 9.0
    checks = {"no falls": not everyone, f"tilt under {MAX_TILT_DEG:.0f} deg (worst {worst_tilt:.1f})": worst_tilt < MAX_TILT_DEG,
              f"speed error under {MAX_SPEED_ERROR} m/s ({speed:.2f})": speed < MAX_SPEED_ERROR,
              f"turn error under {MAX_TURN_ERROR} rad/s ({turn:.2f})": turn < MAX_TURN_ERROR}
    for check, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {check}", flush=True)
    print("RESULT: " + ("PASSED" if all(checks.values()) else "FAILED"), flush=True)


def describe(command: tuple[float, float, float]) -> str:
    forward, _, turn = command
    if forward > 0:
        return f"forward {forward} m/s"
    if forward < 0:
        return f"backward {-forward} m/s"
    if turn:
        return f"turning {turn} rad/s"
    return "standing"


if __name__ == "__main__":
    main()
