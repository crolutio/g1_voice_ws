"""Check the robot's vision in the running walk scene (start_isaac.sh --surroundings, plus vision_node).

Holds up each object in front of the robot's camera (/scene/show) and checks that /vision/scene names it,
counts the greetings for people walking past (/vision/person_arrived), and saves a few camera images.

    source /opt/ros/jazzy/setup.bash && python3 check_vision_in_scene.py --out-dir ~/g1_voice_ws/log/vision
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

OBJECTS = ["apple", "banana", "mug", "bottle", "scissors"]
SHOW_S = 6.0  # wall-clock seconds each object is held up
WATCH_S = 40.0  # then watch people walking past for this long


class VisionChecker(Node):
    def __init__(self, out_dir: str) -> None:
        super().__init__("vision_checker")
        self.out_dir = out_dir
        self.show_pub = self.create_publisher(String, "/scene/show", 10)
        self.scenes: list[tuple[float, str]] = []
        self.greetings = 0
        self.image: np.ndarray | None = None
        self.create_subscription(String, "/vision/scene", lambda m: self.scenes.append((time.monotonic(), m.data)), 10)
        self.create_subscription(String, "/vision/person_arrived", self._on_person, 10)
        self.create_subscription(Image, "/robot/camera/image_raw", self._on_image, 2)

    def _on_person(self, _msg: String) -> None:
        self.greetings += 1

    def _on_image(self, msg: Image) -> None:
        self.image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

    def wait(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def save(self, name: str) -> None:
        if self.image is None:
            return
        from PIL import Image as PilImage

        PilImage.fromarray(self.image).save(os.path.join(self.out_dir, f"{name}.png"))

    def seen_since(self, since: float) -> list[str]:
        return [scene for at, scene in self.scenes if at >= since]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", default=os.path.expanduser("~/g1_voice_ws/log/vision"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rclpy.init()
    checker = VisionChecker(args.out_dir)
    print("Waiting for the camera and the vision node...", flush=True)
    while checker.image is None or not checker.scenes:
        checker.wait(0.5)
    checker.save("view")
    passed = 0
    for name in OBJECTS:
        checker.show_pub.publish(String(data=name))
        since = time.monotonic() + 1.0  # give it a moment to appear and be seen
        checker.wait(SHOW_S)
        checker.save(f"shown_{name}")
        scenes = checker.seen_since(since)
        found = any(name in scene for scene in scenes)
        passed += found
        print(f"{'PASS' if found else 'FAIL'}  showed {name:8s} -> robot saw: {scenes[-1] if scenes else '(nothing)'}",
              flush=True)
    checker.show_pub.publish(String(data="none"))
    before = checker.greetings
    people_seen = 0
    start = time.monotonic()
    while time.monotonic() - start < WATCH_S:
        checker.wait(2.0)
        if checker.scenes and "person" in checker.scenes[-1][1] or (checker.scenes and "people" in checker.scenes[-1][1]):
            people_seen += 1
            if people_seen == 1:
                checker.save("person")
    print(f"{'PASS' if checker.greetings > before else 'FAIL'}  people walking past: greeted {checker.greetings - before} "
          f"time(s) in {WATCH_S:.0f} s", flush=True)
    print(f"RESULT: {passed}/{len(OBJECTS)} objects named; images in {args.out_dir}", flush=True)
    checker.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
