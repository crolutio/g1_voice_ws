#!/usr/bin/env bash
# Terminal 1: start Isaac Sim with the Unitree R1. The trained walking policy moves the legs; the ROS 2
# interface lets the assistant (terminal 2) walk the robot and move its arms, waist and head:
#   in:  /cmd_vel (geometry_msgs/Twist), /robot/joint_command (sensor_msgs/JointState, arms, waist, head)
#   out: /clock, /robot/odom (nav_msgs/Odometry), /robot/joint_states
# Extra arguments go to the scene script, e.g. --viewer kit (full Isaac Sim window, slow), --headless
# or --checkpoint PATH (a specific trained policy).
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source "$HOME/isaacsim-env/bin/activate"
cd "$HOME/g1_voice_ws/training"
exec python -u walk_scene.py "$@"
