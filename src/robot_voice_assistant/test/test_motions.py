import math

import pytest

from robot_voice_assistant.motions import MOTION_DESCRIPTIONS, MotionController, Pose2D


def _run(controller: MotionController, start: Pose2D, max_s: float = 30.0, dt: float = 0.05) -> tuple[Pose2D, float]:
    """Integrate the commanded velocities as a perfect robot would follow them; return the end pose and time."""
    pose, t = start, 0.0
    while t < max_s:
        command = controller.update(pose, t)
        if controller.finished:
            return pose, t
        heading = pose.yaw
        dx = (command.forward * math.cos(heading) - command.sideways * math.sin(heading)) * dt
        dy = (command.forward * math.sin(heading) + command.sideways * math.cos(heading)) * dt
        pose = Pose2D(pose.x + dx, pose.y + dy, pose.yaw + command.turn * dt)
        t += dt
    return pose, t


def ideal() -> MotionController:
    """A controller for a robot that stops instantly (no coasting), as simulated by _run."""
    return MotionController(walk_coast_s=0.0, turn_coast_s=0.0)


def test_every_motion_is_described() -> None:
    assert {"none", "stop", "walk_forward", "walk_backward", "turn_left", "turn_right"} <= set(MOTION_DESCRIPTIONS)


def test_side_steps_are_not_offered() -> None:
    # The trained R1 policy does not walk sideways (0.25 m/s commanded gave 0.00 to 0.07 m/s).
    assert "step_left" not in MOTION_DESCRIPTIONS and "step_right" not in MOTION_DESCRIPTIONS


def test_walk_backward_goes_back_half_a_metre() -> None:
    controller = ideal()
    controller.start("walk_backward", Pose2D(0.0, 0.0, 0.0), now_s=0.0)
    end, _ = _run(controller, Pose2D(0.0, 0.0, 0.0))
    assert end.x == pytest.approx(-0.5, abs=0.05)


def test_controller_stops_early_to_allow_for_coasting() -> None:
    controller = MotionController(walk_coast_s=0.5, turn_coast_s=0.33)
    controller.start("walk_forward", Pose2D(0.0, 0.0, 0.0), now_s=0.0)
    end, _ = _run(controller, Pose2D(0.0, 0.0, 0.0))
    assert end.x == pytest.approx(0.75, abs=0.05)  # 0.5 m/s x 0.5 s: the real robot coasts the last 0.25 m
    controller.start("walk_backward", Pose2D(0.0, 0.0, 0.0), now_s=0.0)
    end, _ = _run(controller, Pose2D(0.0, 0.0, 0.0))
    assert end.x == pytest.approx(-0.35, abs=0.05)  # slower, so it coasts less: 0.3 m/s x 0.5 s
    controller.start("turn_left", Pose2D(0.0, 0.0, 0.0), now_s=0.0)
    end, _ = _run(controller, Pose2D(0.0, 0.0, 0.0))
    assert math.degrees(end.yaw) == pytest.approx(75.0, abs=4.0)


def test_walk_forward_goes_one_metre_and_stops() -> None:
    controller = ideal()
    controller.start("walk_forward", Pose2D(0.0, 0.0, 0.0), now_s=0.0)
    end, _ = _run(controller, Pose2D(0.0, 0.0, 0.0))
    assert end.x == pytest.approx(1.0, abs=0.05)
    assert abs(end.y) < 0.02


def test_walk_forward_follows_the_heading() -> None:
    controller = ideal()
    start = Pose2D(2.0, 1.0, math.pi / 2)
    controller.start("walk_forward", start, now_s=0.0)
    end, _ = _run(controller, start)
    assert end.x == pytest.approx(2.0, abs=0.05)
    assert end.y == pytest.approx(2.0, abs=0.05)


@pytest.mark.parametrize(("motion", "angle"), [("turn_left", 90.0), ("turn_right", -90.0), ("turn_around", 180.0)])
def test_turns_rotate_by_the_right_angle(motion: str, angle: float) -> None:
    controller = ideal()
    controller.start(motion, Pose2D(0.0, 0.0, 0.0), now_s=0.0)
    end, _ = _run(controller, Pose2D(0.0, 0.0, 0.0))
    assert math.degrees(end.yaw) == pytest.approx(angle, abs=4.0)


def test_turning_works_across_the_plus_minus_pi_boundary() -> None:
    controller = ideal()
    start = Pose2D(0.0, 0.0, math.radians(170))
    controller.start("turn_left", start, now_s=0.0)
    end, _ = _run(controller, start)
    turned = math.degrees(math.atan2(math.sin(end.yaw - start.yaw), math.cos(end.yaw - start.yaw)))
    assert turned == pytest.approx(90.0, abs=4.0)


def test_stop_cancels_a_running_motion() -> None:
    controller = ideal()
    controller.start("walk_forward", Pose2D(0.0, 0.0, 0.0), now_s=0.0)
    assert controller.update(Pose2D(0.2, 0.0, 0.0), 1.0).forward > 0
    controller.start("stop", Pose2D(0.2, 0.0, 0.0), now_s=1.0)
    command = controller.update(Pose2D(0.2, 0.0, 0.0), 1.05)
    assert (command.forward, command.sideways, command.turn) == (0.0, 0.0, 0.0)
    assert controller.finished


def test_keep_walking_continues_until_stopped() -> None:
    controller = ideal()
    controller.start("keep_walking", Pose2D(0.0, 0.0, 0.0), now_s=0.0)
    assert controller.update(Pose2D(5.0, 0.0, 0.0), 10.0).forward > 0
    assert not controller.finished


def test_motion_gives_up_after_its_time_limit() -> None:
    controller = ideal()
    controller.start("walk_forward", Pose2D(0.0, 0.0, 0.0), now_s=0.0)
    # The robot is stuck (pose never changes): the motion must still end.
    for t in range(0, 30):
        controller.update(Pose2D(0.0, 0.0, 0.0), float(t))
    assert controller.finished


def test_unknown_motion_is_rejected() -> None:
    with pytest.raises(KeyError):
        ideal().start("backflip", Pose2D(0.0, 0.0, 0.0), now_s=0.0)
