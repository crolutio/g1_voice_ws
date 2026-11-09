"""GITEX visitors walking the hall: where each one is, which way they face, and which walk frame to show.

A visitor walks a looping route of waypoints at a fixed speed and can pause at a waypoint (for example in front
of the robot). The walk animation advances with the distance walked (one cycle per ``cycle_m`` metres), so the
feet keep pace with the ground instead of sliding; while pausing, the visitor stands (``standing_frame``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    pause_s: float = 0.0  # stand here this long on arrival


class Visitor:
    def __init__(self, route: list[Waypoint], speed: float, cycle_m: float, frames: int, standing_frame: int,
                 start_s: float = 0.0) -> None:
        self.route, self.speed, self.cycle_m = route, speed, cycle_m
        self.frames, self.standing_frame, self.start_s = frames, standing_frame, start_s
        # the legs of the loop (back to the first waypoint at the end), each as (start, end, pause at its end)
        self._legs = [(route[i], route[(i + 1) % len(route)]) for i in range(len(route))]
        self._durations = []
        for a, b in self._legs:
            self._durations.append((math.dist((a.x, a.y), (b.x, b.y)) / speed, b.pause_s))
        self.loop_s = sum(walk + pause for walk, pause in self._durations)
        self._walked_per_loop = sum(walk for walk, _ in self._durations) * speed

    def pose(self, t: float) -> tuple[float, float, float, int]:
        """(x, y, yaw, walk frame) at time ``t`` seconds."""
        t = (t + self.start_s) % self.loop_s
        walked = 0.0  # metres walked so far in this loop
        for (a, b), (walk_s, pause_s) in zip(self._legs, self._durations):
            yaw = math.atan2(b.y - a.y, b.x - a.x)
            if t < walk_s:
                share = t / walk_s if walk_s > 0 else 1.0
                distance = walked + t * self.speed
                return a.x + share * (b.x - a.x), a.y + share * (b.y - a.y), yaw, self._frame(distance)
            t -= walk_s
            walked += walk_s * self.speed
            if t < pause_s:
                return b.x, b.y, yaw, self.standing_frame
            t -= pause_s
        last = self.route[0]
        return last.x, last.y, 0.0, self.standing_frame

    def _frame(self, distance_m: float) -> int:
        return int(round((distance_m / self.cycle_m) * self.frames)) % self.frames
