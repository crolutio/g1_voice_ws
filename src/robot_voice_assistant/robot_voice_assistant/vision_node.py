"""Sees through the head camera and tells the others what is around the robot.

Two detectors: YOLO11 (trained on COCO: 80 everyday things, very sure of them, e.g. person, apple, banana, cup,
bottle, scissors) and YOLO-World (open vocabulary: anything named in ``vocabulary``). Their detections are joined;
COCO labels that make no sense indoors (kite, surfboard, ...) are ignored. When someone has walked up, both look
again at that person, enlarged, and CLIP names what they hold (YOLO calls a red apple in a hand a "sports ball").

in:  /robot/camera/image_raw (sensor_msgs/Image, rgb8)
out: /vision/scene (std_msgs/String): what the robot sees, in words, for the language model's prompt
     /vision/detections (std_msgs/String): the detections as JSON (label, confidence, box)
     /vision/person_arrived (std_msgs/String): "person" when someone walks up and should be greeted
"""

from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from robot_voice_assistant.node_runner import run_node
from robot_voice_assistant.scene_context import (NEVER_HELD, Detection, HeldObjectMemory, PersonGreeter, closer_looks,
                                                 describe_scene, from_closer_look, merge_detections, name_held_things,
                                                 STALL_CONFIRM_FRAMES, people_at_the_stall)

# Things the robot should recognise; YOLO-World finds any of these words, no training needed.
DEFAULT_VOCABULARY = [
    "person", "apple", "banana", "orange", "bottle", "cup", "mug", "bowl", "chair", "table", "laptop",
    "phone", "book", "backpack", "plant", "box", "ball", "drill", "scissors", "teapot", "can",
]
DEFAULT_MODEL = os.path.expanduser("~/g1_voice_ws/models/yolo/yolov8s-worldv2.pt")
DEFAULT_COCO_MODEL = os.path.expanduser("~/g1_voice_ws/models/yolo/yolo11s.pt")
DEFAULT_CLIP_MODEL = os.path.expanduser("~/g1_voice_ws/weights/clip/ViT-B-32.pt")
# COCO labels YOLO11 gives to things in this scene that are not them (a red mug as a sports ball, ...)
CLOSER_LOOK_SIZE = 640  # pixels: the closer look is enlarged to the models' input size
DEFAULT_IGNORE = ["kite", "surfboard", "skateboard", "frisbee", "sports ball", "airplane", "boat", "skis", "snowboard"]


class VisionNode(Node):
    def __init__(self) -> None:
        super().__init__("vision_node")
        self.declare_parameter("model", DEFAULT_MODEL)
        self.declare_parameter("coco_model", DEFAULT_COCO_MODEL)  # "" = YOLO-World only
        self.declare_parameter("coco_confidence", 0.4)
        self.declare_parameter("ignore", DEFAULT_IGNORE)
        self.declare_parameter("vocabulary", DEFAULT_VOCABULARY)
        self.declare_parameter("rate_hz", 3.0)
        self.declare_parameter("confidence", 0.25)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("clip_model", DEFAULT_CLIP_MODEL)  # "" = keep YOLO's names for held things
        self.declare_parameter("look_closer", True)  # look again, enlarged, at the person who walked up

        from ultralytics import YOLO, YOLOWorld  # imported here: loading takes a few seconds

        self._model = YOLOWorld(str(self.get_parameter("model").value))
        coco = str(self.get_parameter("coco_model").value)
        self._coco = YOLO(coco) if coco else None
        self._coco_confidence = float(self.get_parameter("coco_confidence").value)
        self._ignore = {str(label) for label in self.get_parameter("ignore").value}
        self._model.set_classes([str(word) for word in self.get_parameter("vocabulary").value])
        self._confidence = float(self.get_parameter("confidence").value)
        self._device = str(self.get_parameter("device").value)
        self._period = 1.0 / float(self.get_parameter("rate_hz").value)
        self._look_closer = bool(self.get_parameter("look_closer").value)
        clip_model = str(self.get_parameter("clip_model").value)
        self._namer = None
        if clip_model:
            from robot_voice_assistant.object_namer import ClipObjectNamer

            words = [str(w) for w in self.get_parameter("vocabulary").value if w != "person" and w not in NEVER_HELD]
            self._namer = ClipObjectNamer(clip_model, words, self._device)
        self._greeter = PersonGreeter(confirm_frames=STALL_CONFIRM_FRAMES)
        self._held = HeldObjectMemory()
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._scene_pub = self.create_publisher(String, "/vision/scene", 10)
        self._detections_pub = self.create_publisher(String, "/vision/detections", 10)
        self._person_pub = self.create_publisher(String, "/vision/person_arrived", 10)
        self.create_subscription(Image, "/robot/camera/image_raw", self._on_image, 2)
        threading.Thread(target=self._detect_loop, daemon=True).start()
        models = "YOLO11 (COCO) and YOLO-World" if self._coco else "YOLO-World"
        models += ", CLIP naming held things" if self._namer else ""
        self.get_logger().info(f"Vision ready ({models}, {len(self.get_parameter('vocabulary').value)} words).")

    def _on_image(self, msg: Image) -> None:
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)[..., :3]
        with self._lock:
            self._latest = image

    def _detect_loop(self) -> None:
        last_scene = ""
        while True:
            started = time.monotonic()
            with self._lock:
                image, self._latest = self._latest, None
            if image is not None:
                detections = self._held.update(time.monotonic(), self._detect(image))
                scene = describe_scene(detections)
                self._scene_pub.publish(String(data=scene))
                self._detections_pub.publish(String(data=json.dumps([d.__dict__ for d in detections])))
                if scene != last_scene:
                    self.get_logger().info(f"I see: {scene}")
                    last_scene = scene
                if self._greeter.update(time.monotonic(), people_at_the_stall(detections)):
                    self.get_logger().info("A visitor stopped at the stall; greeting them.")
                    self._person_pub.publish(String(data="person"))
            time.sleep(max(0.0, self._period - (time.monotonic() - started)))

    def _detect(self, rgb: np.ndarray) -> list[Detection]:
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        per_model = self._run_models([bgr])[0]
        crops = closer_looks(merge_detections(per_model, self._ignore)) if self._look_closer else []
        height, width = bgr.shape[:2]
        parts = []
        for crop in crops:
            part = bgr[int(crop[1] * height):int(crop[3] * height), int(crop[0] * width):int(crop[2] * width)]
            if min(part.shape[:2]) >= 8:
                import cv2

                scale = CLOSER_LOOK_SIZE / max(part.shape[:2])
                part = cv2.resize(part, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else part
                parts.append((crop, part))
        if parts:  # one batch per model for all the people looked at
            for (crop, _), found in zip(parts, self._run_models([part for _, part in parts])):
                per_model += [from_closer_look(detections, crop) for detections in found]
        if self._namer is not None:
            per_model = name_held_things(per_model, lambda box: self._namer.shares(rgb, box))
        return merge_detections(per_model, self._ignore)

    def _run_models(self, images: list[np.ndarray]) -> list[list[list[Detection]]]:
        """For each image (BGR), the detections of each model."""
        per_image = [[] for _ in images]
        models = [(self._model, self._confidence)] + ([(self._coco, self._coco_confidence)] if self._coco else [])
        for model, confidence in models:
            for found, detections in zip(per_image, self._run(model, images, confidence)):
                found.append(detections)
        return per_image

    def _run(self, model, images: list[np.ndarray], confidence: float) -> list[list[Detection]]:
        results = model.predict(images, conf=confidence, device=self._device, verbose=False)
        per_image = []
        for image, result in zip(images, results):
            height, width = image.shape[:2]
            detections = []
            for box, label, score in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                         result.boxes.conf.tolist()):
                x1, y1, x2, y2 = box
                detections.append(Detection(result.names[int(label)], float(score), (x1 + x2) / 2 / width,
                                            (y2 - y1) / height, (x1 / width, y1 / height, x2 / width, y2 / height)))
            per_image.append(detections)
        return per_image


def main() -> None:
    run_node(VisionNode)


if __name__ == "__main__":
    main()
