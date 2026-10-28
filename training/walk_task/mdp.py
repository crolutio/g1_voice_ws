"""Custom terms for R1 walking: move the arms and waist through gesture poses while walking, and rewards
for walking with bent knees."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .reward_math import drift_while_standing, knee_straightness, swing_foot_height_shortfall

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
