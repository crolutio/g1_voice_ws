import pytest

from robot_voice_assistant.timing import StepTimer, seconds_since


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_steps_are_measured_in_order_with_a_total() -> None:
    clock = FakeClock()
    timer = StepTimer(clock)
    with timer.step("knowledge search"):
        clock.now += 0.031
    with timer.step("language model"):
        clock.now += 1.42
    assert [name for name, _ in timer.steps] == ["knowledge search", "language model"]
    assert [seconds for _, seconds in timer.steps] == pytest.approx([0.031, 1.42])
    assert timer.summary() == "knowledge search 0.03 s, language model 1.42 s, total 1.45 s"


def test_steps_that_took_no_time_are_left_out_of_the_summary() -> None:
    clock = FakeClock()
    timer = StepTimer(clock)
    with timer.step("language model"):
        clock.now += 1.0
    with timer.step("shortening"):
        pass
    assert timer.summary() == "language model 1.00 s, total 1.00 s"


def test_a_step_that_fails_is_still_measured() -> None:
    clock = FakeClock()
    timer = StepTimer(clock)
    try:
        with timer.step("language model"):
            clock.now += 2.0
            raise ConnectionError("down")
    except ConnectionError:
        pass
    assert timer.summary() == "language model 2.00 s, total 2.00 s"


def test_seconds_since_a_recent_moment() -> None:
    assert seconds_since(10.0, now=13.25) == 3.25


def test_seconds_since_ignores_unknown_or_stale_moments() -> None:
    assert seconds_since(None, now=13.0) is None
    assert seconds_since(10.0, now=100.0, max_age_s=60.0) is None  # an old moment belongs to an earlier request
    assert seconds_since(20.0, now=13.0) is None  # in the future: belongs to a newer request
