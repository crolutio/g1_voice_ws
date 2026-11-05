"""Walking on flat ground for the Unitree R1 EDU (26 DOF, the robot of the voice assistant scene).

Based on Isaac Lab's Isaac-Velocity-Flat-G1 task, with these changes:
- the Unitree R1 (robot_cfg.py) instead of the G1;
- PhysX physics, as in the voice assistant scene;
- the policy controls only the 12 leg joints; the waist, arms and head are free for gestures and are
  moved through the gesture poses during training, so the policy learns to balance while gesturing;
- rewards for bent knees and lifted feet (the first policy walked stiff-legged on its knee stops), and for
  standing still when told to stop;
- a step clock and rewards for walking foot by foot to it (no jumping, one foot ahead of the other);
- a human gait: bent knee in swing, straight knee at landing, foot landing in front (behind going backward), heel first,
  legs no wider than the hips, and each foot stepping past the other;
- direct turning-rate commands (no heading control) and more standing-still commands, matching voice
  commands such as "walk forward", "turn left" and "stop".
"""

from __future__ import annotations

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
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
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")  # left first, as the step clock expects
# the shoulder joints (shoulder_roll, where the arms hang) are 27.6 cm apart in Unitree's R1 URDF
SHOULDER_WIDTH_M = 0.276
STEP_PERIOD_S = 0.8  # one full step cycle (left and right step), as in Unitree's G1 training


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

        # rewards: walk with bent knees and lifted feet. Without these the first policy locked its knees
        # straight on their -10 degree stop (free support, no penalty) and shuffled with hips and ankles.
        self.rewards.dof_pos_limits.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint", ".*_knee_joint"]
        )
        self.rewards.swing_foot_too_low = RewTerm(
            func=walk_mdp.swing_foot_too_low,
            weight=-20.0,
            params={
                "target_height": 0.08,
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            },
        )

        # rewards: stand still when told to stop. The knee-bending policy kept creeping forward at about
        # 0.25 m/s for seconds after a stop (locked straight knees had stood still by themselves).
        self.rewards.drifting_while_standing = RewTerm(
            func=walk_mdp.drifting_while_standing,
            weight=-2.0,
            params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.stand_still = RewTerm(
            func=mdp.stand_still_joint_deviation_l1,
            weight=-0.2,
            params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
        )
        # more starts and stops: a new command every 4 to 8 s instead of every 10 s
        command.resampling_time_range = (4.0, 8.0)

        # step rhythm: the knee policy hopped on both feet (both in the air 26-36% of the time, almost never on
        # one foot). The policy now sees a step clock and is rewarded for walking foot by foot to it, with one
        # foot ahead of the other, and penalised for every moment both feet are in the air.
        feet_contacts = SceneEntityCfg("contact_forces", body_names=[FOOT_BODIES[0], FOOT_BODIES[1]], preserve_order=True)
        self.observations.policy.step_clock = ObsTerm(func=walk_mdp.step_clock, params={"period": STEP_PERIOD_S})
        self.rewards.jumping = RewTerm(func=walk_mdp.jumping, weight=-2.0, params={"sensor_cfg": feet_contacts})
        self.rewards.step_rhythm = RewTerm(
            func=walk_mdp.step_rhythm,
            weight=1.0,
            params={"period": STEP_PERIOD_S, "stance_fraction": 0.6, "command_name": "base_velocity",
                    "sensor_cfg": feet_contacts},
        )
        self.rewards.one_foot_forward = RewTerm(
            func=walk_mdp.one_foot_forward,
            weight=2.0,
            params={"max_length": 0.3, "command_name": "base_velocity", "sensor_cfg": feet_contacts,
                    "asset_cfg": SceneEntityCfg("robot", body_names=[FOOT_BODIES[0], FOOT_BODIES[1]], preserve_order=True)},
        )

        # Weights: the gait rules shape how it walks, but walking at the commanded speed stays the main goal. With
        # 4-5x larger weights the policy chased foot placement, stopped tracking the commands and became unstable.
        # the human gait: the hip swings the leg forward with the knee bent so the foot clears the floor, the knee
        # straightens just before the foot lands in front of the hips (behind them when walking backward), heel
        # first, and the nearly straight leg takes the weight. (This replaces "knees never straighter than
        # 11 degrees", which also kept the standing leg bent; the knee end stop is still penalised above.)
        knees = SceneEntityCfg("robot", joint_names=["left_knee_joint", "right_knee_joint"], preserve_order=True)
        feet = SceneEntityCfg("robot", body_names=[FOOT_BODIES[0], FOOT_BODIES[1]], preserve_order=True)
        self.rewards.swing_knee_bend = RewTerm(
            func=walk_mdp.swing_knee_bend,
            weight=-1.0,
            params={"target": 0.5, "command_name": "base_velocity", "sensor_cfg": feet_contacts, "asset_cfg": knees},
        )
        self.rewards.landing_knee_straight = RewTerm(
            func=walk_mdp.landing_knee_straight,
            weight=-3.0,
            params={"max_bend": 0.25, "sensor_cfg": feet_contacts, "asset_cfg": knees},
        )
        self.rewards.land_in_front = RewTerm(
            func=walk_mdp.land_in_front,
            weight=4.0,
            params={"max_distance": 0.15, "command_name": "base_velocity", "sensor_cfg": feet_contacts,
                    "asset_cfg": feet},
        )
        self.rewards.stance_knee_bend = RewTerm(  # the step-rhythm policy walked in deep lunges (knees to 124 deg)
            func=walk_mdp.stance_knee_bend,
            weight=-1.0,
            params={"max_bend": 0.6, "sensor_cfg": feet_contacts, "asset_cfg": knees},
        )
        self.rewards.stand_still.weight = -1.0  # stand with the feet together when stopped (it stood in a lunge)
        # feet as far apart as the shoulders, always, also when turning and standing (it walked with the feet
        # 37-53 cm apart); and each foot
        # steps past the other: left in front, then right in front (behind, walking backward)
        self.rewards.feet_width = RewTerm(
            func=walk_mdp.feet_width, weight=-8.0,
            params={"min_width": SHOULDER_WIDTH_M - 0.03, "max_width": SHOULDER_WIDTH_M + 0.03, "asset_cfg": feet},
        )
        self.rewards.land_past_other_foot = RewTerm(
            func=walk_mdp.land_past_other_foot,
            weight=4.0,
            params={"max_distance": 0.3, "command_name": "base_velocity", "sensor_cfg": feet_contacts,
                    "asset_cfg": feet},
        )
        self.rewards.land_heel_first = RewTerm(
            func=walk_mdp.land_heel_first,
            weight=2.0,
            params={"max_tilt": 0.15, "sensor_cfg": feet_contacts, "asset_cfg": feet},
        )

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


@configclass
class R1WalkPlanEnvCfg(R1WalkFlatEnvCfg):
    """Walking on a plan: a foot planter and IK give the leg angles (and the arm swing); the policy adds small
    corrections to keep its balance. Rewarded for following the plan, the commanded speed, and an upright torso."""

    def __post_init__(self):
        super().__post_init__()
        from .plan_action import PlanFollowingActionCfg

        self.actions.joint_pos = PlanFollowingActionCfg(asset_name="robot")
        # gestures while walking and standing, like the voice robot: the arm swing rides on them (see
        # _command_upper_body); arms move at about the speed of the gesture node's gestures
        self.events.animate_upper_body.params["step_fraction"] = 0.08

        policy = self.observations.policy
        policy.step_clock = None
        policy.plan_phase = ObsTerm(func=walk_mdp.plan_phase)
        policy.plan_leg_angles = ObsTerm(func=walk_mdp.plan_leg_angles)
        policy.plan_stance = ObsTerm(func=walk_mdp.plan_stance)
        # the policy sees the joints with the commanded gesture taken out (a raised arm looks like a resting one)
        policy.joint_pos.func = walk_mdp.joint_pos_without_gesture
        policy.joint_vel.func = walk_mdp.joint_vel_without_gesture

        # the plan makes the gait, so the hand-made gait rules are off
        for name in ("swing_foot_too_low", "jumping", "step_rhythm", "one_foot_forward", "swing_knee_bend",
                     "landing_knee_straight", "land_in_front", "stance_knee_bend", "feet_width", "land_past_other_foot",
                     "land_heel_first", "stand_still", "feet_air_time", "joint_deviation_hip"):
            setattr(self.rewards, name, None)
        feet_contacts = SceneEntityCfg("contact_forces", body_names=list(FOOT_BODIES), preserve_order=True)
        feet = SceneEntityCfg("robot", body_names=list(FOOT_BODIES), preserve_order=True)
        self.rewards.follow_plan_feet = RewTerm(func=walk_mdp.follow_plan_feet, weight=1.0,
                                                params={"std": 0.05, "asset_cfg": feet})
        self.rewards.follow_plan_contacts = RewTerm(func=walk_mdp.follow_plan_contacts, weight=0.5,
                                                    params={"sensor_cfg": feet_contacts})
        self.rewards.follow_plan_joints = RewTerm(func=walk_mdp.follow_plan_joints, weight=0.5, params={"std": 0.5})
        # keep the upper body straight (the pelvis and torso upright), and little bobbing or rocking
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.ang_vel_xy_l2.weight = -0.1
        # balance, every step: keep the sideways capture point (where it would have to step to stop) between the feet
        self.rewards.capture_point_outside_feet = RewTerm(func=walk_mdp.capture_point_outside_feet, weight=-20.0,
                                                          params={"margin": 0.03, "asset_cfg": feet})
        # end a practice run as soon as balance is lost (tilted over 35 degrees, or the hips low), not only when the
        # torso hits the ground: no time spent flailing, and the fall penalty comes right at the mistake
        self.terminations.lost_balance = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.6})
        self.terminations.hips_too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})
        # the voice assistant starts, stops and reverses often: practise that, with more standing in between
        self.commands.base_velocity.resampling_time_range = (2.0, 4.0)
        self.commands.base_velocity.rel_standing_envs = 0.25
        # turning on the spot is the hardest move for the voice robot: 30% of new commands practise it
        from .commands import TurnPracticeVelocityCommandCfg

        practice = TurnPracticeVelocityCommandCfg()
        for name, value in vars(self.commands.base_velocity).items():
            if name != "class_type":
                setattr(practice, name, value)
        self.commands.base_velocity = practice
        self.commands.base_velocity.rel_turn_in_place_envs = 0.1  # 0.3 made slow turns and walking worse
        # pushed more often than the default (every 10-15 s), so it keeps its balance when bumped
        self.events.push_robot.interval_range_s = (6.0, 10.0)
        # room for bigger balance corrections on top of the plan
        self.actions.joint_pos.scale = 0.35


@configclass
class R1WalkPlanEnvCfg_PLAY(R1WalkPlanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
