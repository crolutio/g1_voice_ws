"""Action term for walking on a plan: foot planter + IK give the leg joint angles, the policy adds corrections.

Every control step, per robot: the step clock advances (faster when walking faster), the walk grows in over
the first step after standing, the foot planter places the feet (planted feet locked to the ground), IK turns
the feet into the 12 leg joint angles, and each arm swings with its own leg (on top of any commanded arm pose). The policy's 12 actions are
small corrections on top of the planned leg angles, so it learns to keep its balance rather than a gait.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from foot_placement.gait_planner import FeetPlan, FootPlanter, GaitPlan, arm_motion, cycle_period, move_toward, needs_to_step, step_walk_in, swing_on_arm_pose  # noqa: E402
from foot_placement.leg_kinematics import LegKinematics  # noqa: E402

from .reward_math import heading_from_quat_xyzw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

URDF = os.path.expanduser("~/g1_voice_ws/robots/r1/R1.urdf")
LEG_JOINT_ORDER = [f"{side}_{name}_joint" for side in ("left", "right")
                   for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
ARM_SWING_JOINTS = ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_elbow_joint", "right_elbow_joint"]


class PlanFollowingAction(JointPositionAction):
    cfg: PlanFollowingActionCfg

    def __init__(self, cfg: PlanFollowingActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._env = env
        self.plan_cfg = GaitPlan()
        # the planner and IK are hundreds of tiny tensor operations: on a GPU with a single robot they are much
        # slower than on the CPU, so they can run on their own device (results are copied to the sim device)
        plan_device = cfg.plan_device or self.device
        self._plan_device = plan_device
        self._legs = [LegKinematics.from_urdf(URDF, side).to(plan_device) for side in ("left", "right")]
        self.planter = FootPlanter(self.plan_cfg, self.num_envs, device=plan_device)
        self._phase = torch.full((self.num_envs,), self.plan_cfg.stance_fraction, device=plan_device)
        self._walk_in = torch.zeros(self.num_envs, device=plan_device)
        standing = torch.cat([leg.default_angles() for leg in self._legs], dim=1)  # (1, 12)
        self._reference = standing.expand(self.num_envs, 12).clone()
        self.phase = self._phase.to(self.device)  # step clock, for observations
        self.reference = self._reference.to(self.device)  # planned leg joint angles
        self.feet = None  # the latest FeetPlan (targets relative to the hips), for rewards and observations
        self._arm_ids = self._asset.find_joints(ARM_SWING_JOINTS, preserve_order=True)[0]
        self._arm_neutral = self._asset.data.default_joint_pos.torch[:, self._arm_ids].clone()
        self.arm_pose = self._arm_neutral.clone()  # arm pose the swing rides on: follows the commanded pose smoothly
        self._arm_goal = self._arm_neutral.clone()  # commanded arm pose (gestures)
        self.arm_targets = self._arm_neutral.clone()

    def process_actions(self, actions: torch.Tensor) -> None:
        dt = self._env.step_dt
        here = self._plan_device
        command = self._env.command_manager.get_command(self.cfg.command_name).to(here)
        plan = self.plan_cfg
        moving = (torch.linalg.norm(command[:, :2], dim=1) > plan.moving_threshold) | (
            torch.abs(command[:, 2]) > plan.moving_threshold
        )
        data = self._asset.data
        velocity = data.root_lin_vel_b.torch[:, :2].to(here)  # robot frame, like the command
        stumbling = needs_to_step(plan, command, velocity) & self.cfg.reactive_stepping  # pushed: step to catch itself
        self._walk_in = step_walk_in(self._walk_in, moving, dt, self.cfg.walk_in_s, stumbling)  # eases in and out
        ramp = 0.5 - 0.5 * torch.cos(torch.pi * self._walk_in)
        self._phase = torch.where(self._walk_in > 0,
                                  torch.remainder(self._phase + dt / cycle_period(plan, command), 1.0),
                                  torch.full_like(self._phase, plan.stance_fraction))
        hips_xy = data.root_pos_w.torch[:, :2].to(here)
        hips_yaw = heading_from_quat_xyzw(data.root_quat_w.torch).to(here)
        hips_velocity = data.root_lin_vel_w.torch[:, :2].to(here)
        feet = self.planter.step(hips_xy, hips_yaw, command, self._phase, ramp,
                                 hips_velocity=hips_velocity if self.cfg.catch_steps else None)
        for index, leg in enumerate(self._legs):
            columns = slice(6 * index, 6 * index + 6)
            self._reference[:, columns] = leg.solve(
                feet.position[:, index], feet.rotation[:, index], self._reference[:, columns],
                iterations=self.cfg.ik_iterations, safe=True,  # a far target never flips the leg
            )
        shoulder, elbow = arm_motion(plan, self._phase, command, ramp)
        swing = torch.cat([shoulder, elbow], dim=1).to(self.device)
        self.arm_pose = move_toward(self.arm_pose, self._arm_goal, self.cfg.arm_speed * dt)  # arms never jump
        self.arm_targets = swing_on_arm_pose(self.arm_pose, self._arm_neutral, swing)
        self.phase = self._phase.to(self.device)
        self.reference = self._reference.to(self.device)
        self.feet = FeetPlan(feet.position.to(self.device), feet.rotation.to(self.device), feet.stance.to(self.device))

        self._raw_actions[:] = actions
        self._processed_actions = self.reference + self._raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def set_arm_pose(self, pose: torch.Tensor) -> None:
        """Commanded arm pose (N, 4) in ARM_SWING_JOINTS order, e.g. from the gesture node; reached smoothly."""
        self._arm_goal[:] = pose

    def apply_actions(self) -> None:
        super().apply_actions()
        self._asset.set_joint_position_target_index(target=self.arm_targets, joint_ids=self._arm_ids)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        everyone = env_ids is None or isinstance(env_ids, slice)
        ids = slice(None) if everyone else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        plan_ids = ids if everyone else ids.to(self._plan_device)
        self.planter.reset(None if everyone else plan_ids)
        self._phase[plan_ids] = self.plan_cfg.stance_fraction
        self._walk_in[plan_ids] = 0.0
        self._reference[plan_ids] = torch.cat([leg.default_angles() for leg in self._legs], dim=1)
        self.phase[ids] = self.plan_cfg.stance_fraction
        self.reference[ids] = self._reference[plan_ids].to(self.device)


@configclass
class PlanFollowingActionCfg(JointPositionActionCfg):
    class_type: type = PlanFollowingAction
    joint_names: list[str] = LEG_JOINT_ORDER
    preserve_order: bool = True
    use_default_offset: bool = False
    scale: float = 0.25  # corrections on top of the plan (rad per action unit)
    command_name: str = "base_velocity"
    walk_in_s: float = 0.8
    ik_iterations: int = 4  # warm-started from the previous step, so a few iterations suffice
    reactive_stepping: bool = True  # start stepping when pushed or stumbling, even when told to stand
    catch_steps: bool = True  # land each step where the body is tipping
    arm_speed: float = 3.0  # rad/s: the fastest a commanded arm pose is followed (a 95 deg drop takes 0.55 s)
    plan_device: str | None = None  # where the planner and IK run (None: the sim device; "cpu" for one robot)
