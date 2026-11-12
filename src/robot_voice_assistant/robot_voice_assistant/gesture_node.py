"""ROS 2 node: plays named gestures as smooth upper-body joint targets for the simulated R1."""

from __future__ import annotations

from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from robot_voice_assistant.gestures import GESTURE_KEYFRAMES, NEUTRAL_POSE, UPPER_BODY_JOINTS, GestureTrajectory, Pose
from robot_voice_assistant.node_runner import run_node
from robot_voice_assistant.safety import SAFE_GESTURE, Person, gesture_allowed, parse_people

PEOPLE_STALE_S = 1.0  # people last reported longer ago than this are not known any more


class GestureNode(Node):
    def __init__(self) -> None:
        super().__init__("gesture_node")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("command_topic", "/robot/joint_command")

        self._current_pose: Pose = dict(NEUTRAL_POSE)
        self._trajectory: GestureTrajectory | None = None
        self._trajectory_start_ns = 0
        self._people: list[Person] = []
        self._people_ns = -(10 ** 18)

        self._command_pub = self.create_publisher(JointState, str(self.get_parameter("command_topic").value), 10)
        self._active_pub = self.create_publisher(String, "/robot/active_gesture", 10)
        self.create_subscription(String, "/robot/gesture", self._on_gesture, 10)
        self.create_subscription(String, "/robot/nearby_people", self._on_people, 10)
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self._on_tick)
        self.get_logger().info(f"Gestures available: {', '.join(sorted(GESTURE_KEYFRAMES))}")

    def _on_people(self, msg: String) -> None:
        self._people, self._people_ns = parse_people(msg.data), self.get_clock().now().nanoseconds

    def _on_gesture(self, msg: String) -> None:
        gesture = msg.data.strip().lower()
        if gesture not in GESTURE_KEYFRAMES:
            self.get_logger().warn(f"Unknown gesture '{gesture}'")
            return
        if gesture == "none" and self._trajectory is not None:
            return  # let a running gesture finish
        fresh = (self.get_clock().now().nanoseconds - self._people_ns) / 1e9 <= PEOPLE_STALE_S
        if not gesture_allowed(gesture, self._people if fresh else []):
            self.get_logger().warn(f"Someone is too close for '{gesture}'; doing '{SAFE_GESTURE}' instead.")
            gesture = SAFE_GESTURE
        self._trajectory = GestureTrajectory(gesture, self._current_pose)
        self._trajectory_start_ns = self.get_clock().now().nanoseconds
        self._active_pub.publish(String(data=gesture))
        self.get_logger().info(f"Playing '{gesture}' ({self._trajectory.duration_s:.1f}s)")

    def _on_tick(self) -> None:
        if self._trajectory is not None:
            elapsed_s = (self.get_clock().now().nanoseconds - self._trajectory_start_ns) / 1e9
            self._current_pose = self._trajectory.sample(elapsed_s)
            if self._trajectory.is_finished(elapsed_s):
                self._trajectory = None
                self._active_pub.publish(String(data="none"))

        command = JointState()
        command.header.stamp = self.get_clock().now().to_msg()
        command.name = list(UPPER_BODY_JOINTS)
        command.position = [self._current_pose[joint] for joint in UPPER_BODY_JOINTS]
        self._command_pub.publish(command)


def main() -> None:
    run_node(GestureNode)


if __name__ == "__main__":
    main()
