"""Named walking movements ("walk forward", "turn left", ...) executed in closed loop on odometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

NO_MOTION = "none"


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class VelocityCommand:
    forward: float = 0.0  # m/s, in the robot's frame
    sideways: float = 0.0  # m/s, positive to the left
    turn: float = 0.0  # rad/s, positive to the left


@dataclass(frozen=True)
class MotionSpec:
    forward_m: float = 0.0  # distance along the starting heading (negative = backwards)
    sideways_m: float = 0.0  # distance to the left of the starting heading (negative = right)
    turn_deg: float = 0.0  # positive = left
    speed: float = 0.5  # m/s for walking, rad/s for turning
    continuous: bool = False  # keep going until another motion or "stop"


# There is no "walk_backward": the trained walking policy does not walk backwards (it stands still instead).
MOTIONS: dict[str, MotionSpec] = {
    "walk_forward": MotionSpec(forward_m=1.0, speed=0.5),
    "step_left": MotionSpec(sideways_m=0.3, speed=0.2),
    "step_right": MotionSpec(sideways_m=-0.3, speed=0.2),
    "turn_left": MotionSpec(turn_deg=90.0, speed=0.8),
    "turn_right": MotionSpec(turn_deg=-90.0, speed=0.8),
    "turn_around": MotionSpec(turn_deg=180.0, speed=0.8),
    "keep_walking": MotionSpec(speed=0.5, continuous=True),
}

MOTION_DESCRIPTIONS: dict[str, str] = {
    "none": "stay where you are",
    "stop": "stop walking",
    "walk_forward": "walk forward about one metre, e.g. 'come here' or 'walk forward'",
    "step_left": "take a small step to the left",
    "step_right": "take a small step to the right",
    "turn_left": "turn 90 degrees to the left",
    "turn_right": "turn 90 degrees to the right",
    "turn_around": "turn around (180 degrees)",
    "keep_walking": "keep walking forward until told to stop",
}

POSITION_TOLERANCE_M = 0.03
ANGLE_TOLERANCE_RAD = math.radians(2.0)
SLOWDOWN_DISTANCE_M = 0.25  # slow down over the last part of a walk
SLOWDOWN_ANGLE_RAD = math.radians(25.0)
MIN_SPEED_FRACTION = 0.5  # the walking policy barely responds to very small commands
TIME_LIMIT_FACTOR = 3.0  # give up after 3x the nominal duration (robot stuck or pushed)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class MotionController:
    """Turns one named motion into velocity commands until the odometry says it is done."""

    def __init__(self) -> None:
        self._spec: MotionSpec | None = None
        self._start = Pose2D(0.0, 0.0, 0.0)
        self._deadline_s = 0.0
        self._last_yaw = 0.0
        self._turned_rad = 0.0  # accumulated rotation since the motion started, so 180 degree turns work
        self.name = NO_MOTION

    @property
    def finished(self) -> bool:
        return self._spec is None

    def start(self, name: str, pose: Pose2D, now_s: float) -> None:
        if name in (NO_MOTION, "stop"):
            self._spec, self.name = None, NO_MOTION
            return
        spec = MOTIONS[name]
        nominal_s = max(
            math.hypot(spec.forward_m, spec.sideways_m) / spec.speed if spec.forward_m or spec.sideways_m else 0.0,
            math.radians(abs(spec.turn_deg)) / spec.speed if spec.turn_deg else 0.0,
        )
        self._spec, self._start, self.name = spec, pose, name
        self._last_yaw, self._turned_rad = pose.yaw, 0.0
        self._deadline_s = math.inf if spec.continuous else now_s + TIME_LIMIT_FACTOR * nominal_s + 2.0

    def update(self, pose: Pose2D, now_s: float) -> VelocityCommand:
        spec = self._spec
        if spec is None:
            return VelocityCommand()
        if now_s > self._deadline_s:
            self._spec, self.name = None, NO_MOTION
            return VelocityCommand()
        if spec.continuous:
            return VelocityCommand(forward=spec.speed)

        if spec.turn_deg:
            self._turned_rad += _wrap_angle(pose.yaw - self._last_yaw)
            self._last_yaw = pose.yaw
            remaining = math.radians(spec.turn_deg) - self._turned_rad
            if abs(remaining) <= ANGLE_TOLERANCE_RAD:
                self._spec, self.name = None, NO_MOTION
                return VelocityCommand()
            scale = max(MIN_SPEED_FRACTION, min(1.0, abs(remaining) / SLOWDOWN_ANGLE_RAD))
            return VelocityCommand(turn=math.copysign(spec.speed * scale, remaining))

        # Walking: target point fixed in the world, expressed in the robot's current frame.
        cos_s, sin_s = math.cos(self._start.yaw), math.sin(self._start.yaw)
        target_x = self._start.x + spec.forward_m * cos_s - spec.sideways_m * sin_s
        target_y = self._start.y + spec.forward_m * sin_s + spec.sideways_m * cos_s
        dx_world, dy_world = target_x - pose.x, target_y - pose.y
        cos_p, sin_p = math.cos(pose.yaw), math.sin(pose.yaw)
        ahead = dx_world * cos_p + dy_world * sin_p
        left = -dx_world * sin_p + dy_world * cos_p
        distance = math.hypot(ahead, left)
        if distance <= POSITION_TOLERANCE_M:
            self._spec, self.name = None, NO_MOTION
            return VelocityCommand()
        speed = spec.speed * max(MIN_SPEED_FRACTION, min(1.0, distance / SLOWDOWN_DISTANCE_M))
        # Hold the starting heading while walking.
        heading_error = _wrap_angle(self._start.yaw - pose.yaw)
        return VelocityCommand(forward=speed * ahead / distance, sideways=speed * left / distance, turn=1.5 * heading_error)


# Example phrases for recognising walking commands by meaning (semantic.CommandMatcher).
MOTION_PHRASES: dict[str, list[str]] = {
    "walk_forward": ["come here", "come over to me", "walk towards me", "walk forward", "move forward"],
    "keep_walking": ["keep walking", "keep going", "walk until I say stop"],
    "stop": ["stop", "stop walking", "halt", "stand still"],
    "turn_left": ["turn left", "turn to your left", "rotate left"],
    "turn_right": ["turn right", "turn to your right", "rotate right"],
    "turn_around": ["turn around", "face the other way", "turn back"],
    "step_left": ["step to the left", "move a little to the left", "side step left"],
    "step_right": ["step to the right", "move a little to the right", "side step right"],
}
