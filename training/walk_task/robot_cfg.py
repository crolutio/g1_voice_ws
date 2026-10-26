"""Isaac Lab articulation config for the Unitree R1 EDU (26 DOF, no hands).

The USD was converted from Unitree's r1_description/R1.urdf with Isaac Lab's convert_urdf.py
(--merge_joints). Effort and velocity limits come from the URDF; the joint stiffness and damping
follow a community R1 walking setup (Syszhao/Unitree_R1_WALK, Apache-2.0), with stiffer waist and
arms so gestures are followed closely.
"""

from __future__ import annotations

import os

from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

R1_USD_PATH = os.path.expanduser("~/g1_voice_ws/robots/r1/usd/r1.usd/R1/R1.usda")

# Standing pose: knees slightly bent with the soles flat (checked with forward kinematics), arms
# relaxed as in the gesture library's neutral pose.
R1_STANDING_JOINT_POS = {
    ".*_hip_pitch_joint": -0.25,
    ".*_knee_joint": 0.50,
    ".*_ankle_pitch_joint": -0.25,
    ".*_shoulder_pitch_joint": 0.15,
    "left_shoulder_roll_joint": 0.12,
    "right_shoulder_roll_joint": -0.12,
    ".*_elbow_joint": 1.3,
}

R1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=R1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos=R1_STANDING_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=18.8,
            stiffness={".*_hip_pitch_joint": 80.0, ".*_hip_roll_joint": 80.0, ".*_hip_yaw_joint": 60.0, ".*_knee_joint": 80.0},
            damping={".*_hip_pitch_joint": 3.0, ".*_hip_roll_joint": 3.0, ".*_hip_yaw_joint": 2.0, ".*_knee_joint": 4.0},
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=30.0,
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=18.7,
            stiffness=60.0,
            damping=3.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_roll_joint"],
            effort_limit_sim={".*_shoulder_pitch_joint": 60.0, ".*_shoulder_roll_joint": 60.0,
                              ".*_shoulder_yaw_joint": 33.0, ".*_elbow_joint": 33.0, ".*_wrist_roll_joint": 33.0},
            velocity_limit_sim=18.7,
            stiffness={".*_shoulder_.*_joint": 60.0, ".*_elbow_joint": 40.0, ".*_wrist_roll_joint": 20.0},
            damping={".*_shoulder_.*_joint": 3.0, ".*_elbow_joint": 2.0, ".*_wrist_roll_joint": 1.0},
            armature=0.01,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_.*_joint"],
            effort_limit_sim={"head_pitch_joint": 33.0, "head_yaw_joint": 0.86},
            velocity_limit_sim=11.0,
            stiffness={"head_pitch_joint": 15.0, "head_yaw_joint": 2.0},
            damping={"head_pitch_joint": 1.0, "head_yaw_joint": 0.1},
            armature=0.001,
        ),
    },
)
