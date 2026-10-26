"""Upper-body gesture library for the Unitree R1 (26 DOF, no hands) and smooth trajectory playback.

Gestures use the arms (5 joints each), the waist (roll and yaw) and the neck (head pitch and yaw).
The R1 has no waist pitch, so a bow is a bow of the head. Angles are joint position targets in radians.

Joint conventions, measured with forward kinematics on Unitree's R1 URDF:
- elbow: 0 = forearm bent 90 degrees pointing forward, +1.57 = straight, negative = bent further.
- shoulder_pitch: negative raises the arm forward (about -1.5 = horizontal, -2.7 = overhead).
- shoulder_roll: moves the arm out to the side; positive on the left arm, negative on the right.
- shoulder_yaw: positive turns the left forearm outward and the right forearm inward.
- head_pitch: positive looks down (nod). head_yaw and waist_yaw: positive turns to the left.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

Pose = dict[str, float]
Keyframes = list[tuple[float, Pose]]

# Joint limits in degrees, from Unitree's R1 URDF (r1_description/R1.urdf).
JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "waist_roll_joint": (-30.0, 30.0),
    "waist_yaw_joint": (-150.0, 150.0),
    "left_shoulder_pitch_joint": (-180.0, 120.0),
    "left_shoulder_roll_joint": (-13.0, 142.0),
    "left_shoulder_yaw_joint": (-110.0, 110.0),
    "left_elbow_joint": (-55.9, 125.2),
    "left_wrist_roll_joint": (-110.0, 110.0),
    "right_shoulder_pitch_joint": (-180.0, 120.0),
    "right_shoulder_roll_joint": (-142.0, 13.0),
    "right_shoulder_yaw_joint": (-110.0, 110.0),
    "right_elbow_joint": (-55.9, 125.2),
    "right_wrist_roll_joint": (-110.0, 110.0),
    "head_pitch_joint": (-36.0, 36.0),
    "head_yaw_joint": (-115.0, 115.0),
}
UPPER_BODY_JOINTS: tuple[str, ...] = tuple(JOINT_LIMITS_DEG)

# Arms hanging relaxed beside the hips with slightly bent elbows, head looking straight ahead.
NEUTRAL_POSE: Pose = {joint: 0.0 for joint in UPPER_BODY_JOINTS} | {
    "left_shoulder_pitch_joint": 0.15,
    "left_shoulder_roll_joint": 0.12,
    "left_elbow_joint": 1.3,
    "right_shoulder_pitch_joint": 0.15,
    "right_shoulder_roll_joint": -0.12,
    "right_elbow_joint": 1.3,
}

RETURN_DURATION_S = 0.9


def _pose(**overrides: float) -> Pose:
    """A full pose: the neutral pose with some joints changed (pass names without `_joint`)."""
    pose = dict(NEUTRAL_POSE)
    for short_name, angle in overrides.items():
        joint = f"{short_name}_joint"
        if joint not in pose:
            raise KeyError(joint)
        pose[joint] = angle
    return pose


def _both_arms(pitch: float, roll_out: float, elbow: float, yaw_out: float = 0.0, **extra: float) -> Pose:
    """Mirror one arm pose onto both arms; roll_out/yaw_out are positive away from the body."""
    return _pose(
        left_shoulder_pitch=pitch, left_shoulder_roll=roll_out, left_shoulder_yaw=yaw_out, left_elbow=elbow,
        right_shoulder_pitch=pitch, right_shoulder_roll=-roll_out, right_shoulder_yaw=-yaw_out, right_elbow=elbow,
        **extra,
    )


def _hold(pose: Pose, start_s: float, hold_s: float) -> Keyframes:
    return [(start_s, pose), (start_s + hold_s, pose)]


def _wave(side: str) -> Keyframes:
    """Hand raised beside the head with the forearm up, swinging side to side (never across the face)."""
    outward = 1.0 if side == "left" else -1.0
    raised = {f"{side}_shoulder_pitch": -1.4, f"{side}_shoulder_roll": outward * 1.0, f"{side}_elbow": 0.0}
    swing_in = _pose(**raised, **{f"{side}_shoulder_yaw": -outward * 0.15})
    swing_out = _pose(**raised, **{f"{side}_shoulder_yaw": outward * 0.55})
    keyframes: Keyframes = [(0.8, _pose(**raised))]
    t = 0.8
    for _ in range(3):
        keyframes += [(t + 0.3, swing_in), (t + 0.6, swing_out)]
        t += 0.6
    keyframes.append((t + 0.3, _pose(**raised)))
    return keyframes


def _alternate(first: Pose, second: Pose, start_s: float, period_s: float, count: int) -> Keyframes:
    keyframes: Keyframes = []
    t = start_s
    for _ in range(count):
        keyframes += [(t, first), (t + period_s / 2, second)]
        t += period_s
    return keyframes


GESTURE_DESCRIPTIONS: dict[str, str] = {
    "none": "no movement, just stand relaxed",
    "wave_right": "wave with the right arm; for greetings, goodbyes, or when asked to wave",
    "wave_left": "wave with the left arm, when asked to wave with the left arm or hand",
    "nod": "nod your head yes; when agreeing, confirming, or giving a simple factual answer",
    "shake": "shake your head no; when disagreeing or saying something is not true",
    "bow": "a polite bow of the head with the arms together in front; when thanked or for a formal greeting",
    "point_forward": "point straight ahead with the right arm; when asked to point or showing a direction ahead",
    "celebrate": "raise both arms in celebration; for good news, success or excitement",
    "shrug": "shrug; when you do not know, are unsure, or did not understand",
    "think": "arm raised to the chin with the head tilted; when explaining something or giving advice",
    "open_arms": "open both arms wide; for a warm welcome or talking about something big",
    "look_left": "turn the head and upper body to the left; when asked to look left",
    "look_right": "turn the head and upper body to the right; when asked to look right",
}

_ARMS_UP_NARROW = _both_arms(pitch=-2.7, roll_out=0.55, elbow=1.35, head_pitch=-0.2)
_ARMS_UP_WIDE = _both_arms(pitch=-2.7, roll_out=0.85, elbow=1.35, head_pitch=-0.2)
_SHRUG = _both_arms(pitch=-0.2, roll_out=0.35, elbow=0.1, yaw_out=0.6, head_pitch=-0.1, waist_roll=0.05)
# Hand about 9 cm in front of the chin; found by a search that keeps the arm 6.7 cm from the head.
_THINK = _pose(right_shoulder_pitch=-0.8, right_shoulder_roll=-0.2, right_shoulder_yaw=0.45, right_elbow=-0.75,
               head_pitch=0.1, head_yaw=-0.1)
_POINT = _pose(right_shoulder_pitch=-1.5, right_shoulder_roll=-0.1, right_elbow=1.5, head_yaw=-0.05)
_BOW = _both_arms(pitch=-0.35, roll_out=0.05, elbow=0.6, yaw_out=-0.35, head_pitch=0.5)

GESTURE_KEYFRAMES: dict[str, Keyframes] = {
    "none": [],
    "wave_right": _wave("right"),
    "wave_left": _wave("left"),
    "nod": _alternate(_pose(head_pitch=0.35), _pose(head_pitch=-0.05), 0.35, 0.7, 2),
    "shake": _alternate(_pose(head_yaw=0.5), _pose(head_yaw=-0.5), 0.4, 0.8, 2),
    "bow": _hold(_BOW, 0.9, 1.0),
    "point_forward": _hold(_POINT, 0.8, 1.4),
    "celebrate": [(1.0, _ARMS_UP_NARROW)] + _alternate(_ARMS_UP_WIDE, _ARMS_UP_NARROW, 1.4, 0.8, 2),
    "shrug": _hold(_SHRUG, 0.6, 1.0),
    "think": [(0.9, _THINK), (2.4, _THINK | {"head_yaw_joint": 0.1})],
    "open_arms": _hold(_both_arms(pitch=-0.5, roll_out=1.2, elbow=1.3, head_pitch=-0.1), 0.9, 1.2),
    "look_left": _hold(_pose(head_yaw=0.8, waist_yaw=0.3), 0.8, 1.3),
    "look_right": _hold(_pose(head_yaw=-0.8, waist_yaw=-0.3), 0.8, 1.3),
}


def _ease(fraction: float) -> float:
    """Cosine ease-in/ease-out so joints start and stop smoothly."""
    return 0.5 - 0.5 * math.cos(math.pi * min(max(fraction, 0.0), 1.0))


class GestureTrajectory:
    """Time-parameterized path: current pose -> gesture keyframes -> neutral pose."""

    def __init__(self, gesture: str, start_pose: Mapping[str, float]) -> None:
        keyframes = GESTURE_KEYFRAMES[gesture]
        start = {joint: float(start_pose.get(joint, NEUTRAL_POSE[joint])) for joint in UPPER_BODY_JOINTS}
        last_time = keyframes[-1][0] if keyframes else 0.0
        self._waypoints: Keyframes = [(0.0, start), *keyframes, (last_time + RETURN_DURATION_S, dict(NEUTRAL_POSE))]
        self.gesture = gesture
        self.duration_s = self._waypoints[-1][0]

    def is_finished(self, elapsed_s: float) -> bool:
        return elapsed_s >= self.duration_s

    def sample(self, elapsed_s: float) -> Pose:
        if elapsed_s >= self.duration_s:
            return dict(self._waypoints[-1][1])
        for (t0, pose0), (t1, pose1) in zip(self._waypoints, self._waypoints[1:]):
            if elapsed_s <= t1:
                fraction = _ease((elapsed_s - t0) / (t1 - t0)) if t1 > t0 else 1.0
                return {joint: pose0[joint] + fraction * (pose1[joint] - pose0[joint]) for joint in UPPER_BODY_JOINTS}
        return dict(self._waypoints[-1][1])


# Example phrases for recognising explicit gesture requests by meaning (semantic.CommandMatcher).
GESTURE_PHRASES: dict[str, list[str]] = {
    "wave_right": ["wave at me", "wave your hand", "wave hello"],
    "wave_left": ["wave with your left hand", "wave your left hand"],
    "nod": ["nod your head", "nod"],
    "shake": ["shake your head", "shake your head no"],
    "bow": ["bow", "take a bow", "bow down"],
    "point_forward": ["point forward", "point ahead", "point straight ahead"],
    "celebrate": ["celebrate", "raise your arms", "put your hands up"],
    "shrug": ["shrug", "shrug your shoulders"],
    "think": ["show me your thinking pose", "think about it with your hand on your chin"],
    "open_arms": ["open your arms", "spread your arms wide"],
    "look_left": ["look left", "look to your left"],
    "look_right": ["look right", "look to your right"],
}
