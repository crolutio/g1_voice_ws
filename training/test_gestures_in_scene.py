"""Test the running walk scene with the real gesture node: play each gesture standing, then walking.

Needs start_isaac.sh and the gesture node (ros2 run robot_voice_assistant gesture_node --ros-args -p
use_sim_time:=true) running. Reports, per gesture, the lowest pelvis height; below 0.5 m the robot fell.

    source /opt/ros/jazzy/setup.bash && python3 test_gestures_in_scene.py
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

GESTURES = ["wave_right", "wave_left", "nod", "bow", "celebrate", "open_arms", "point_forward", "shrug", "think",
            "shake", "look_left", "look_right"]
GESTURE_SECONDS = 5.0  # simulated seconds per gesture (the longest takes about 4 s)
FALLEN_BELOW_M = 0.5


class GestureTester(Node):
    def __init__(self) -> None:
        super().__init__("gesture_tester")
        self.gesture_pub = self.create_publisher(String, "/robot/gesture", 10)
        self.velocity_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.height: float | None = None
        self.sim_time = 0.0
        self.create_subscription(Odometry, "/robot/odom", self._on_odom, 10)
        self.create_subscription(Clock, "/clock", self._on_clock, 10)

    def _on_odom(self, msg: Odometry) -> None:
        self.height = msg.pose.pose.position.z

    def _on_clock(self, msg: Clock) -> None:
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def run(self, gesture: str, forward: float) -> float:
        self.gesture_pub.publish(String(data=gesture))
        start, lowest = self.sim_time, self.height
        while self.sim_time - start < GESTURE_SECONDS:
            velocity = Twist()
            velocity.linear.x = forward
            self.velocity_pub.publish(velocity)
            rclpy.spin_once(self, timeout_sec=0.02)
            lowest = min(lowest, self.height)
        return lowest


def main() -> None:
    rclpy.init()
    tester = GestureTester()
    while tester.height is None or tester.sim_time == 0.0:
        rclpy.spin_once(tester, timeout_sec=0.1)
    for forward, label in ((0.0, "standing"), (0.4, "walking 0.4 m/s")):
        for gesture in GESTURES:
            lowest = tester.run(gesture, forward)
            fell = lowest < FALLEN_BELOW_M
            print(f"{label:16s} {gesture:14s} pelvis lowest {lowest:4.2f} m{'  FELL' if fell else ''}", flush=True)
            if fell:  # the scene puts a fallen robot back on its feet; let it settle before the next gesture
                tester.run("none", 0.0)
    tester.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
