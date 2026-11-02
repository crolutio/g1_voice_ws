"""Custom terms for R1 walking: move the arms and waist through gesture poses while walking, and rewards
for walking with bent knees."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .reward_math import (
    both_feet_in_air,
    drift_while_standing,
    gait_phase_match,
    heading_from_quat_xyzw,
    heel_first,
    knee_straightness,
    landing_in_front,
    landing_knee_excess,
    landing_passes_other_foot,
    sideways_capture_point_outside,
    stance_knee_excess,
    step_width_error,
    stride_length,
    swing_foot_height_shortfall,
    swing_knee_shortfall,
    toe_up_from_quat_xyzw,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

_STATE_ATTRIBUTE = "_upper_body_state"


class _UpperBodyState:
    def __init__(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, pose_file: str, animated_fraction: float):
        with open(pose_file, encoding="utf-8") as file:
            data = json.load(file)
        asset = env.scene[asset_cfg.name]
        joint_ids, _ = asset.find_joints(data["joint_names"], preserve_order=True)
        device = env.device
        self.joint_ids = list(joint_ids)
        self.poses = torch.tensor(data["poses"], dtype=torch.float32, device=device)
        self.neutral = torch.tensor(data["neutral"], dtype=torch.float32, device=device)
        self.goal = self.neutral.repeat(env.num_envs, 1)
        self.target = self.neutral.repeat(env.num_envs, 1)
        # Some robots keep their arms still, so the policy also sees plain walking.
        self.animated = torch.rand(env.num_envs, device=device) < animated_fraction


def _state(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, pose_file: str, animated_fraction: float):
    state = getattr(env, _STATE_ATTRIBUTE, None)
    if state is None:
        state = _UpperBodyState(env, asset_cfg, pose_file, animated_fraction)
        setattr(env, _STATE_ATTRIBUTE, state)
    return state


def _as_index_tensor(env: ManagerBasedEnv, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    return env_ids.to(env.device)


def reset_upper_body(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    pose_file: str,
    animated_fraction: float = 0.7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """On reset: arms and waist back to the neutral pose."""
    state = _state(env, asset_cfg, pose_file, animated_fraction)
    ids = _as_index_tensor(env, env_ids)
    state.goal[ids] = state.neutral
    state.target[ids] = state.neutral
    env.scene[asset_cfg.name].actuators.target_command.set_position_index(
        value=state.target[ids], joint_ids=state.joint_ids, env_ids=ids
    )


def animate_upper_body(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    pose_file: str,
    animated_fraction: float = 0.7,
    resample_probability: float = 0.02,
    step_fraction: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Every policy step: move the arm and waist targets a little towards a randomly chosen gesture pose."""
    state = _state(env, asset_cfg, pose_file, animated_fraction)
    ids = _as_index_tensor(env, env_ids)
    resample = ids[(torch.rand(len(ids), device=env.device) < resample_probability) & state.animated[ids]]
    if len(resample) > 0:
        choice = torch.randint(len(state.poses), (len(resample),), device=env.device)
        state.goal[resample] = state.poses[choice]
    state.target[ids] += step_fraction * (state.goal[ids] - state.target[ids])
    env.scene[asset_cfg.name].actuators.target_command.set_position_index(
        value=state.target[ids], joint_ids=state.joint_ids, env_ids=ids
    )


# --- rewards for walking with bent knees (see reward_math.py) ---


def _moving(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return (torch.linalg.norm(command[:, :2], dim=1) > 0.1) | (torch.abs(command[:, 2]) > 0.1)


def knees_too_straight(env: ManagerBasedRLEnv, min_bend: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalise knees straighter than ``min_bend`` (rad); the R1 knee stop is at -0.17 rad."""
    asset = env.scene[asset_cfg.name]
    return knee_straightness(asset.data.joint_pos.torch[:, asset_cfg.joint_ids], min_bend)


def swing_foot_too_low(
    env: ManagerBasedRLEnv,
    target_height: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalise a swinging foot that is lifted less than ``target_height`` above the standing foot."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0
    asset = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    return swing_foot_height_shortfall(foot_z, in_contact, target_height) * _moving(env, command_name)


def drifting_while_standing(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalise body motion while the robot is told to stand still (so "stop" means stop)."""
    asset = env.scene[asset_cfg.name]
    velocity = torch.cat([asset.data.root_lin_vel_b.torch[:, :2], asset.data.root_ang_vel_b.torch[:, 2:3]], dim=1)
    return drift_while_standing(velocity, env.command_manager.get_command(command_name))


# --- step rhythm: a step clock the policy sees, and rewards for walking foot by foot (see reward_math.py) ---


def _step_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    return torch.remainder(env.episode_length_buf.float() * env.step_dt / period, 1.0)


def _feet_in_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene.sensors[sensor_cfg.name].data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0


def step_clock(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """Observation: where in the step cycle the robot is, as (sin, cos)."""
    angle = 2.0 * torch.pi * _step_phase(env, period)
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)


def jumping(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalise moments with both feet in the air."""
    return both_feet_in_air(_feet_in_contact(env, sensor_cfg))


def step_rhythm(
    env: ManagerBasedRLEnv, period: float, stance_fraction: float, command_name: str, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet that follow the step clock (left, both, right, both, ...); both down when standing."""
    return gait_phase_match(
        _feet_in_contact(env, sensor_cfg), _step_phase(env, period), _moving(env, command_name), stance_fraction
    )


def one_foot_forward(
    env: ManagerBasedRLEnv, max_length: float, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward one foot ahead of the other (along the heading) while both are down and walking forward or back."""
    asset = env.scene[asset_cfg.name]
    yaw = heading_from_quat_xyzw(asset.data.root_quat_w.torch)
    offset = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :2] - asset.data.root_pos_w.torch[:, None, :2]
    feet_x = offset[..., 0] * torch.cos(yaw)[:, None] + offset[..., 1] * torch.sin(yaw)[:, None]
    walking = torch.abs(env.command_manager.get_command(command_name)[:, 0]) > 0.1
    return stride_length(_feet_in_contact(env, sensor_cfg), feet_x, walking, max_length)


# --- the human gait (see reward_math.py): bent knee in swing, straight knee at landing, foot in front, heel first ---


def _heading_yaw(asset) -> torch.Tensor:
    return heading_from_quat_xyzw(asset.data.root_quat_w.torch)  # Isaac Lab 3.0 quaternions are (x, y, z, w)


def _first_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]


def swing_knee_bend(
    env: ManagerBasedRLEnv, target: float, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalise a swinging leg whose knee is bent less than ``target`` (rad): the foot must clear the floor."""
    knees = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids]
    return swing_knee_shortfall(knees, _feet_in_contact(env, sensor_cfg), _moving(env, command_name), target)


def landing_knee_straight(
    env: ManagerBasedRLEnv, max_bend: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalise landing on a bent knee: the knee straightens just before the foot touches down."""
    knees = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids]
    return landing_knee_excess(knees, _first_contact(env, sensor_cfg), max_bend)


def land_in_front(
    env: ManagerBasedRLEnv, max_distance: float, command_name: str, sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward a foot landing in front of the hips (behind them when walking backward)."""
    asset = env.scene[asset_cfg.name]
    yaw = _heading_yaw(asset)
    offset = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :2] - asset.data.root_pos_w.torch[:, None, :2]
    feet_x = offset[..., 0] * torch.cos(yaw)[:, None] + offset[..., 1] * torch.sin(yaw)[:, None]
    forward = env.command_manager.get_command(command_name)[:, 0]
    direction = torch.where(forward > 0.1, 1.0, torch.where(forward < -0.1, -1.0, 0.0))
    return landing_in_front(feet_x, _first_contact(env, sensor_cfg), direction, max_distance)


def land_heel_first(env: ManagerBasedRLEnv, max_tilt: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet that touch down with the toes slightly up (heel first)."""
    toe_up = toe_up_from_quat_xyzw(env.scene[asset_cfg.name].data.body_quat_w.torch[:, asset_cfg.body_ids])
    return heel_first(toe_up, _first_contact(env, sensor_cfg), max_tilt)


def stance_knee_bend(env: ManagerBasedRLEnv, max_bend: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalise a deeply bent knee on a leg that carries weight (no lunging or crouching)."""
    knees = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids]
    return stance_knee_excess(knees, _feet_in_contact(env, sensor_cfg), max_bend)


def _feet_in_heading_frame(asset, body_ids) -> tuple[torch.Tensor, torch.Tensor]:
    """Feet relative to the hips: (along the heading, to the robot's left), in metres."""
    yaw = _heading_yaw(asset)
    offset = asset.data.body_pos_w.torch[:, body_ids, :2] - asset.data.root_pos_w.torch[:, None, :2]
    cos, sin = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    return offset[..., 0] * cos + offset[..., 1] * sin, -offset[..., 0] * sin + offset[..., 1] * cos


def capture_point_outside_feet(env: ManagerBasedRLEnv, margin: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalise a sideways capture point outside the feet: the robot moving sideways faster than it can stop."""
    asset = env.scene[asset_cfg.name]
    yaw = _heading_yaw(asset)
    velocity = asset.data.root_lin_vel_w.torch[:, :2]
    sideways = -velocity[:, 0] * torch.sin(yaw) + velocity[:, 1] * torch.cos(yaw)
    _, feet_y = _feet_in_heading_frame(asset, asset_cfg.body_ids)
    return sideways_capture_point_outside(sideways, asset.data.root_pos_w.torch[:, 2], feet_y, margin)


def feet_width(env: ManagerBasedRLEnv, min_width: float, max_width: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalise feet further apart (or closer together) side to side than the hips allow."""
    _, feet_y = _feet_in_heading_frame(env.scene[asset_cfg.name], asset_cfg.body_ids)
    return step_width_error(feet_y, min_width, max_width)


def land_past_other_foot(
    env: ManagerBasedRLEnv, max_distance: float, command_name: str, sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward a landing foot that is past the other foot (ahead walking forward, behind walking backward)."""
    feet_x, _ = _feet_in_heading_frame(env.scene[asset_cfg.name], asset_cfg.body_ids)
    forward = env.command_manager.get_command(command_name)[:, 0]
    direction = torch.where(forward > 0.1, 1.0, torch.where(forward < -0.1, -1.0, 0.0))
    return landing_passes_other_foot(feet_x, _first_contact(env, sensor_cfg), direction, max_distance)


# --- walking on a plan (plan_action.py): what the policy sees, and how closely it follows the plan ---


def _plan(env: ManagerBasedRLEnv):
    return env.action_manager.get_term("joint_pos")


def plan_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation: where in the step cycle the plan is, as (sin, cos)."""
    angle = 2.0 * torch.pi * _plan(env).phase
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)


def plan_leg_angles(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation: the planned leg joint angles, relative to the default standing pose."""
    term = _plan(env)
    return term.reference - env.scene["robot"].data.default_joint_pos.torch[:, term._joint_ids]


def plan_stance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation: which feet the plan wants on the ground (1) or swinging (0)."""
    feet = _plan(env).feet
    return feet.stance.float() if feet is not None else torch.ones(env.num_envs, 2, device=env.device)


def follow_plan_feet(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet close to where the plan puts them (relative to the hips, in the heading frame)."""
    feet = _plan(env).feet
    if feet is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    feet_x, feet_y = _feet_in_heading_frame(asset, asset_cfg.body_ids)
    feet_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2] - asset.data.root_pos_w.torch[:, None, 2]
    error = (feet_x - feet.position[..., 0]) ** 2 + (feet_y - feet.position[..., 1]) ** 2 + (feet_z - feet.position[..., 2]) ** 2
    return torch.exp(-error / std**2).mean(dim=1)


def follow_plan_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet that are on the ground when the plan says planted, and in the air when it says swinging."""
    feet = _plan(env).feet
    if feet is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (_feet_in_contact(env, sensor_cfg) == feet.stance).float().mean(dim=1)


def follow_plan_joints(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Reward leg joint angles close to the planned (IK) angles."""
    term = _plan(env)
    angles = env.scene["robot"].data.joint_pos.torch[:, term._joint_ids]
    return torch.exp(-torch.sum((angles - term.reference) ** 2, dim=1) / std**2)
