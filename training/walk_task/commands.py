"""Velocity commands with turning-on-the-spot practice (the voice robot's turn left, right and around)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

from .reward_math import turn_in_place


class TurnPracticeVelocityCommand(UniformVelocityCommand):
    cfg: TurnPracticeVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        chosen = torch.rand(len(ids), device=self.device) < self.cfg.rel_turn_in_place_envs
        low, high = self.cfg.turn_in_place_rate
        self.vel_command_b[ids] = turn_in_place(
            self.vel_command_b[ids], chosen, low, high, torch.rand(len(ids), device=self.device)
        )
        self.is_standing_env[ids] &= ~chosen


@configclass
class TurnPracticeVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = TurnPracticeVelocityCommand
    rel_turn_in_place_envs: float = 0.3  # share of new commands that are turning on the spot
    turn_in_place_rate: tuple[float, float] = (0.4, 1.0)  # rad/s
