"""Heading hold for the walk scene: when told to walk straight, keep facing the way the robot faced at the start.

The policy is given a turn rate, not a heading, so small turn-rate errors add up while walking. This remembers the
heading when a straight walk starts and adds a small, limited turn command that steers back to it.
"""

from __future__ import annotations

import math


class HeadingHold:
    def __init__(self, gain: float = 1.5, max_turn: float = 0.4, threshold: float = 0.05) -> None:
        self.gain = gain  # rad/s of turn per rad of heading error
        self.max_turn = max_turn  # rad/s, the largest correction
        self.threshold = threshold  # commands smaller than this count as zero
        self._target: float | None = None

    def turn_command(self, command: tuple[float, float, float], yaw: float) -> float:
        """The turn rate (rad/s) to send for ``command`` (forward m/s, left m/s, turn rad/s) at heading ``yaw``."""
        forward, left, turn = command
        if abs(turn) > self.threshold or math.hypot(forward, left) <= self.threshold:
            self._target = None  # turning or standing: nothing to hold; a new straight walk starts from here
            return turn
        if self._target is None:
            self._target = yaw
        error = math.remainder(self._target - yaw, 2 * math.pi)
        return max(-self.max_turn, min(self.max_turn, self.gain * error))
