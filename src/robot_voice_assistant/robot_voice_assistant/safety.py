"""Keeping a safe distance from people while walking, turning and gesturing.

People are given in the robot's frame (x forward, y to the left, metres), as a range sensor around the robot
would report them; in the simulation the scene publishes them on /robot/nearby_people. The robot never walks
closer than ``PERSONAL_SPACE_M`` to anyone (centre to centre: its own body, a person's body and room to spare),
turns on the spot only with nobody within arm's reach, and holds its arms out only with room for them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

from robot_voice_assistant.motions import MOTIONS, VelocityCommand

Person = tuple[float, float]

PERSONAL_SPACE_M = 1.0  # closest the robot walks to a person, centre to centre
STOP_MARGIN_M = 0.25  # stop this much before the personal space: the walk coasts on after the command stops
TURN_CLEARANCE_M = 0.6  # the arms swing out a little while turning on the spot
GESTURE_CLEARANCE_M = 0.8  # an arm held out reaches about 0.6 m from the robot's centre
# Gestures that hold an arm (or the upper body) out towards the person; the others stay close to the body.
WIDE_GESTURES = frozenset({"wave_right", "wave_left", "point_forward", "open_arms", "celebrate", "bow"})
SAFE_GESTURE = "nod"  # done instead of a wide gesture when someone stands too close


def room_to_walk(people: Sequence[Person], forward: bool, personal_space_m: float = PERSONAL_SPACE_M) -> float:
    """How far the robot can walk straight ahead (or back) before coming closer than ``personal_space_m`` to
    anyone; 0 when someone is already that close on the way, infinite when nobody is in the way."""
    sign = 1.0 if forward else -1.0
    room = math.inf
    for x, y in people:
        along = sign * x
        if along <= 0.0 or abs(y) >= personal_space_m:
            continue  # behind the direction of travel, or passed at a safe distance
        room = min(room, max(0.0, along - math.sqrt(personal_space_m ** 2 - y ** 2)))
    return room


def _nearest(people: Sequence[Person]) -> float:
    return min((math.hypot(x, y) for x, y in people), default=math.inf)


def turn_allowed(people: Sequence[Person]) -> bool:
    return _nearest(people) >= TURN_CLEARANCE_M


def gesture_allowed(gesture: str, people: Sequence[Person]) -> bool:
    return gesture not in WIDE_GESTURES or _nearest(people) >= GESTURE_CLEARANCE_M


def safe_command(command: VelocityCommand, people: Sequence[Person]) -> tuple[VelocityCommand, str | None]:
    """The walking command with any movement towards someone too close taken out, and why (None: unchanged)."""
    forward, turn, reasons = command.forward, command.turn, []
    if forward != 0.0 and room_to_walk(people, forward=forward > 0) < STOP_MARGIN_M:
        forward = 0.0
        reasons.append(f"a person is within {PERSONAL_SPACE_M:.0f} m {'ahead' if command.forward > 0 else 'behind'}")
    if turn != 0.0 and not turn_allowed(people):
        turn = 0.0
        reasons.append("a person is within arm's reach")
    if not reasons:
        return command, None
    sideways = command.sideways if forward or turn else 0.0
    return VelocityCommand(forward, sideways, turn), "; ".join(reasons)


def parse_people(text: str) -> list[Person]:
    """/robot/nearby_people: a JSON list of [x, y] in the robot's frame; anything else means nobody is known."""
    try:
        people = json.loads(text)
        return [(float(x), float(y)) for x, y in people]
    except (ValueError, TypeError):
        return []


def motion_blocked(name: str, people: Sequence[Person]) -> str | None:
    """Why the named motion must not start now (None: it may). Stopping is always allowed."""
    spec = MOTIONS.get(name)
    if spec is None:
        return None
    walking = spec.forward_m != 0.0 or spec.continuous
    direction = spec.forward_m if spec.forward_m != 0.0 else spec.speed
    command = VelocityCommand(forward=math.copysign(spec.speed, direction) if walking else 0.0,
                              turn=spec.speed if spec.turn_deg else 0.0)
    return safe_command(command, people)[1]
