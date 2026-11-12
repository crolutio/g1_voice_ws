"""Named walking movements ("walk forward", "turn left", ...) executed in closed loop on odometry."""

from __future__ import annotations

import math
import re
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
    time_limit_s: float | None = None  # give up after this long (default: from the distance and speed)


# There are no side steps: the trained R1 walking policy does not walk sideways (measured 0.00 to 0.07 m/s
# for 0.25 m/s commanded), so "step left" or "step right" would not move the robot.
MOTIONS: dict[str, MotionSpec] = {
    "walk_forward": MotionSpec(forward_m=1.0, speed=0.5),
    # The walking policy was not trained to walk backward: it goes back at about 0.04 m/s for 0.3 m/s commanded
    # (visitor test, 2026-11-11), so half a metre takes 15 s or more, slowing down at the end included.
    "walk_backward": MotionSpec(forward_m=-0.5, speed=0.3, time_limit_s=20.0),
    # Turning on the spot at 0.5 rad/s: at 0.8 rad/s the walk fell in 13 of 32 robots after a few minutes of use
    # (test_falls.py "voice robot sequence x3"); at 0.5 rad/s none fell. A 90 degree turn takes about 3 s.
    "turn_left": MotionSpec(turn_deg=90.0, speed=0.5),
    "turn_right": MotionSpec(turn_deg=-90.0, speed=0.5),
    "turn_around": MotionSpec(turn_deg=180.0, speed=0.5),
    "keep_walking": MotionSpec(speed=0.5, continuous=True),
}

MOTION_DESCRIPTIONS: dict[str, str] = {
    "none": "stay where you are",
    "stop": "stop walking",
    "walk_forward": "walk forward about one metre, e.g. 'come here' or 'walk forward'",
    "walk_backward": "walk backward about half a metre",
    "turn_left": "turn 90 degrees to the left",
    "turn_right": "turn 90 degrees to the right",
    "turn_around": "turn around (180 degrees)",
    "keep_walking": "keep walking forward until told to stop",
}

# The robot keeps moving for a moment after the command stops; measured with the final R1 policy it
# coasted about 0.3 m after walking at 0.5 m/s and 13 to 22 degrees after turning at 0.8 rad/s. The
# controller stops early by the distance (or angle) covered in this time at the motion's speed.
DEFAULT_WALK_COAST_S = 0.5
DEFAULT_TURN_COAST_S = 0.33
POSITION_TOLERANCE_M = 0.03
ANGLE_TOLERANCE_RAD = math.radians(2.0)
SLOWDOWN_DISTANCE_M = 0.25  # slow down over the last part of a walk
SLOWDOWN_ANGLE_RAD = math.radians(25.0)
MIN_SPEED_FRACTION = 0.5  # the walking policy barely responds to very small commands
# While turning on the spot, the walking policy drifts (about 0.3 m backward over a turn-around); the controller
# walks against it, gently, so the robot ends where it started (and keeps its distance from people around it).
TURN_HOLD_GAIN = 3.0  # m/s per metre of drift
TURN_HOLD_MAX_SPEED = 0.15
TIME_LIMIT_FACTOR = 3.0  # give up after 3x the nominal duration (robot stuck or pushed)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class MotionController:
    """Turns one named motion into velocity commands until the odometry says it is done."""

    def __init__(
        self,
        walk_coast_s: float = DEFAULT_WALK_COAST_S,
        turn_coast_s: float = DEFAULT_TURN_COAST_S,
    ) -> None:
        self._walk_coast_s = walk_coast_s
        self._turn_coast_s = turn_coast_s
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
        limit_s = spec.time_limit_s if spec.time_limit_s is not None else TIME_LIMIT_FACTOR * nominal_s + 2.0
        self._deadline_s = math.inf if spec.continuous else now_s + limit_s

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
            if abs(remaining) <= ANGLE_TOLERANCE_RAD + spec.speed * self._turn_coast_s:
                self._spec, self.name = None, NO_MOTION
                return VelocityCommand()
            scale = max(MIN_SPEED_FRACTION, min(1.0, abs(remaining) / SLOWDOWN_ANGLE_RAD))
            # hold the place it turns on: walk back against any drift along its current heading
            ahead = (self._start.x - pose.x) * math.cos(pose.yaw) + (self._start.y - pose.y) * math.sin(pose.yaw)
            hold = max(-TURN_HOLD_MAX_SPEED, min(TURN_HOLD_MAX_SPEED, TURN_HOLD_GAIN * ahead))
            return VelocityCommand(forward=hold, turn=math.copysign(spec.speed * scale, remaining))

        # Walking: target point fixed in the world, expressed in the robot's current frame.
        cos_s, sin_s = math.cos(self._start.yaw), math.sin(self._start.yaw)
        target_x = self._start.x + spec.forward_m * cos_s - spec.sideways_m * sin_s
        target_y = self._start.y + spec.forward_m * sin_s + spec.sideways_m * cos_s
        dx_world, dy_world = target_x - pose.x, target_y - pose.y
        cos_p, sin_p = math.cos(pose.yaw), math.sin(pose.yaw)
        ahead = dx_world * cos_p + dy_world * sin_p
        left = -dx_world * sin_p + dy_world * cos_p
        distance = math.hypot(ahead, left)
        if distance <= POSITION_TOLERANCE_M + spec.speed * self._walk_coast_s:
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
    "walk_backward": ["walk backwards", "step back", "move back a little", "back up"],
}


# "Stop" must stop the robot at once, not after the language model answers (with Groq rate-limited, the local model
# took 13 s while the robot kept walking). A command: at the start of what was said or of a clause ("Okay R1, stop").
_STOP = re.compile(r"(?:^|[,.!?;]\s*|\b(?:please|okay|ok|now|just|hey|r1|robot)\s+)(?:stop|halt|freeze)\b"
                   r"|\bstand still\b|\bstop (?:walking|moving)\b", re.IGNORECASE)
_NOT_STOP = re.compile(r"\b(?:don't|dont|do not|never)\s+stop\b", re.IGNORECASE)


def is_stop_request(text: str) -> bool:
    """Whether the person tells the robot to stop moving."""
    return bool(_STOP.search(text.strip())) and not _NOT_STOP.search(text)
