"""Measuring how long each step of answering takes, for the timing lines in the logs."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

# Steps shorter than this (e.g. a check that found nothing to do) are left out of the summary.
MIN_REPORTED_S = 0.005


class StepTimer:
    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self.steps: list[tuple[str, float]] = []

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        started = self._clock()
        try:
            yield
        finally:
            self.steps.append((name, self._clock() - started))

    @property
    def total_s(self) -> float:
        return sum(seconds for _, seconds in self.steps)

    def summary(self) -> str:
        parts = [f"{name} {seconds:.2f} s" for name, seconds in self.steps if seconds >= MIN_REPORTED_S]
        return ", ".join([*parts, f"total {self.total_s:.2f} s"])


def seconds_since(moment: float | None, now: float | None = None, max_age_s: float = 60.0) -> float | None:
    """Seconds from ``moment`` (wall-clock time) until ``now``; None if unknown, in the future or too old."""
    if moment is None:
        return None
    elapsed = (time.time() if now is None else now) - moment
    return elapsed if 0.0 <= elapsed <= max_age_s else None
