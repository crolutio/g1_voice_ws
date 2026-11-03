"""Test the running walk scene (start_isaac.sh) over ROS 2, the way the voice assistant drives it.

Sends /cmd_vel and a resting arm pose on /robot/joint_command, and reports for each step of the test how far
the robot went (in simulation time, from /clock), how far it turned, the lowest pelvis height (below about
0.5 m it fell) and how much each shoulder swung.

    source /opt/ros/jazzy/setup.bash && python3 drive_walk_scene.py
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState

ARM_JOINTS = ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_elbow_joint", "right_elbow_joint"]
REST_SHOULDER, REST_ELBOW = 0.15, 1.3  # the gesture node's resting arm pose
FALLEN_BELOW_M = 0.5
TESTS = [  # (label, forward m/s, turn rad/s, simulated seconds, right shoulder pitch or None for resting)
    ("stand", 0.0, 0.0, 2.0, None),
    ("walk forward 0.5 m/s", 0.5, 0.0, 6.0, None),
    ("stop", 0.0, 0.0, 2.0, None),
    ("walk backward 0.3 m/s", -0.3, 0.0, 5.0, None),
    ("stop", 0.0, 0.0, 2.0, None),
    ("turn left 0.5 rad/s", 0.0, 0.5, 6.4, None),
    ("turn right 0.5 rad/s", 0.0, -0.5, 6.4, None),
    ("walk forward 0.5 m/s again", 0.5, 0.0, 5.0, None),
    ("stop", 0.0, 0.0, 3.0, None),
]


class WalkSceneDriver(Node):
    def __init__(self) -> None:
        super().__init__("walk_scene_driver")
        self.velocity_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.arms_pub = self.create_publisher(JointState, "/robot/joint_command", 10)
        self.odom: Odometry | None = None
        self.sim_time = 0.0
        self.joints: dict[str, float] = {}
        self.create_subscription(Odometry, "/robot/odom", self._on_odom, 10)
        self.create_subscription(Clock, "/clock", self._on_clock, 10)
        self.create_subscription(JointState, "/robot/joint_states", self._on_joints, 10)

    def _on_odom(self, msg: Odometry) -> None:
        self.odom = msg

    def _on_clock(self, msg: Clock) -> None:
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _on_joints(self, msg: JointState) -> None:
        self.joints.update(zip(msg.name, msg.position))

    def pose(self) -> tuple[float, float, float, float]:
        position, q = self.odom.pose.pose.position, self.odom.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return position.x, position.y, position.z, yaw

    def run(self, label: str, forward: float, turn: float, seconds: float, right_shoulder: float | None) -> bool:
        start, start_time = self.pose(), self.sim_time
        lowest = start[2]
        swing = {"left": [], "right": []}
        while self.sim_time - start_time < seconds:
            velocity = Twist()
            velocity.linear.x, velocity.angular.z = forward, turn
            self.velocity_pub.publish(velocity)
            arms = JointState()
            arms.name = ARM_JOINTS
            arms.position = [REST_SHOULDER, REST_SHOULDER if right_shoulder is None else right_shoulder,
                             REST_ELBOW, REST_ELBOW]
            self.arms_pub.publish(arms)
            rclpy.spin_once(self, timeout_sec=0.02)
            lowest = min(lowest, self.pose()[2])
            for side, values in swing.items():
                if f"{side}_shoulder_pitch_joint" in self.joints:
                    values.append(self.joints[f"{side}_shoulder_pitch_joint"])
        end = self.pose()
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        turned = math.degrees(math.remainder(end[3] - start[3], 2 * math.pi))
        swung = {side: math.degrees(max(v) - min(v)) if v else 0.0 for side, v in swing.items()}
        fell = lowest < FALLEN_BELOW_M
        print(f"{label:32s} {distance:4.2f} m ({distance / seconds:4.2f} m/s)  turned {turned:6.1f} deg  "
              f"pelvis lowest {lowest:4.2f} m  shoulder swing L {swung['left']:3.0f} R {swung['right']:3.0f} deg"
              f"{'  FELL' if fell else ''}", flush=True)
        return not fell


def main() -> None:
    rclpy.init()
    driver = WalkSceneDriver()
    while driver.odom is None or driver.sim_time == 0.0:
        rclpy.spin_once(driver, timeout_sec=0.1)
    for test in TESTS:
        if not driver.run(*test):
            print("The robot fell; stopping the test.", flush=True)
            break
    driver.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
