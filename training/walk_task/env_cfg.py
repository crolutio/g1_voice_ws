"""Walking on flat ground for the Unitree R1 EDU (26 DOF, the robot of the voice assistant scene).

Based on Isaac Lab's Isaac-Velocity-Flat-G1 task, with these changes:
- the Unitree R1 (robot_cfg.py) instead of the G1;
- PhysX physics, as in the voice assistant scene;
- the policy controls only the 12 leg joints; the waist, arms and head are free for gestures and are
  moved through the gesture poses during training, so the policy learns to balance while gesturing;
- direct turning-rate commands (no heading control) and more standing-still commands, matching voice
  commands such as "walk forward", "turn left" and "stop".
"""

from __future__ import annotations

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.core.velocity import mdp
from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

from . import mdp as walk_mdp
from .robot_cfg import R1_CFG

POSE_FILE = os.path.join(os.path.dirname(__file__), "upper_body_poses.json")
LEG_JOINTS = [".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", ".*_ankle_.*_joint"]
TORSO_BODY = "waist_yaw_link"
BASE_BODY = "pelvis_link"


@configclass
class R1WalkFlatEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # physics: PhysX, like the voice assistant scene
        self.sim.physics.default = self.sim.physics.isaacsim_physx

        # robot: the R1 with relaxed arms
        self.scene.robot = replace(R1_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        # body names of the R1 (the G1 task refers to the G1's torso_link)
        self.terminations.base_contact.params["sensor_cfg"].body_names = [TORSO_BODY, BASE_BODY, "head_.*_link"]
        self.events.add_base_mass.params["asset_cfg"].body_names = BASE_BODY
        self.events.base_external_force_torque.params["asset_cfg"].body_names = BASE_BODY
        self.commands.base_velocity.marker_pos_offset = (0.0, 0.0, 0.7)

        # actions: legs only
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=LEG_JOINTS, scale=0.5, use_default_offset=True
        )

        # commands: forward/backward, sideways and turning; 20% of the robots are told to stand still
        command = self.commands.base_velocity
        command.heading_command = False
        command.rel_standing_envs = 0.2
        command.ranges.lin_vel_x = (-0.5, 1.0)
        command.ranges.lin_vel_y = (-0.3, 0.3)
        command.ranges.ang_vel_z = (-1.0, 1.0)

        # rewards: the arms, waist and head are not controlled by the policy, so they are not penalised
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_torso = None
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)

        # events: gesture-like arm and waist motion
        self.events.reset_upper_body = EventTerm(
            func=walk_mdp.reset_upper_body, mode="reset", params={"pose_file": POSE_FILE}
        )
        self.events.animate_upper_body = EventTerm(
            func=walk_mdp.animate_upper_body,
            mode="interval",
            interval_range_s=(self.sim.dt * self.decimation, self.sim.dt * self.decimation),
            params={"pose_file": POSE_FILE},
        )


@configclass
class R1WalkFlatEnvCfg_PLAY(R1WalkFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
