"""ROS 2 node: executes named walking motions (/robot/motion) as /cmd_vel commands, closed loop on /robot/odom."""

from __future__ import annotations

import math

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from robot_voice_assistant.motions import MOTIONS, NO_MOTION, MotionController, Pose2D
from robot_voice_assistant.node_runner import run_node


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class MotionNode(Node):
    def __init__(self) -> None:
        super().__init__("motion_node")
        self.declare_parameter("rate_hz", 20.0)
        self._controller = MotionController()
        self._pose: Pose2D | None = None
        self._was_moving = False

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._active_pub = self.create_publisher(String, "/robot/active_motion", 10)
        self.create_subscription(Odometry, "/robot/odom", self._on_odom, 10)
        self.create_subscription(String, "/robot/motion", self._on_motion, 10)
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self._on_tick)
        self.get_logger().info(f"Motions available: stop, {', '.join(sorted(MOTIONS))}")

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _on_odom(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self._pose = Pose2D(p.x, p.y, _yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def _on_motion(self, msg: String) -> None:
        name = msg.data.strip().lower()
        if name not in MOTIONS and name not in (NO_MOTION, "stop"):
            self.get_logger().warn(f"Unknown motion '{name}'")
            return
        if name == NO_MOTION:
            return  # "none" means: keep doing what you are doing
        if self._pose is None:
            self.get_logger().warn("No odometry yet (is the walking scene running?); ignoring the motion.")
            return
        self._controller.start(name, self._pose, self._now_s())
        self._active_pub.publish(String(data=self._controller.name))
        self.get_logger().info(f"Motion: {name}")

    def _on_tick(self) -> None:
        if self._pose is None:
            return
        command = self._controller.update(self._pose, self._now_s())
        moving = not self._controller.finished
        if moving or self._was_moving:
            twist = Twist()
            twist.linear.x, twist.linear.y, twist.angular.z = command.forward, command.sideways, command.turn
            self._cmd_pub.publish(twist)
        if self._was_moving and not moving:
            self._active_pub.publish(String(data=NO_MOTION))
        self._was_moving = moving


def main() -> None:
    run_node(MotionNode)


if __name__ == "__main__":
    main()
