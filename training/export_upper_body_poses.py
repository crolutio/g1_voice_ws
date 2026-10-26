"""Write every gesture pose (arms, waist and head) to walk_task/upper_body_poses.json.

During walking training the upper body is moved through these poses, so the policy learns to keep its
balance while the robot gestures. Run again after changing gestures.py:

    ~/g1_voice_ws/.venv/bin/python ~/g1_voice_ws/training/export_upper_body_poses.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/g1_voice_ws/src/robot_voice_assistant"))
from robot_voice_assistant.gestures import GESTURE_KEYFRAMES, NEUTRAL_POSE, UPPER_BODY_JOINTS  # noqa: E402

poses = [NEUTRAL_POSE] + [pose for keyframes in GESTURE_KEYFRAMES.values() for _, pose in keyframes]
unique = sorted({tuple(round(pose[joint], 4) for joint in UPPER_BODY_JOINTS) for pose in poses})
output = os.path.join(os.path.dirname(__file__), "walk_task", "upper_body_poses.json")
with open(output, "w", encoding="utf-8") as file:
    json.dump({"joint_names": list(UPPER_BODY_JOINTS), "neutral": [NEUTRAL_POSE[j] for j in UPPER_BODY_JOINTS],
               "poses": [list(pose) for pose in unique]}, file, indent=1)
print(f"wrote {len(unique)} poses to {output}")
