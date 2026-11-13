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


def test_a_stop_request_is_recognised_without_the_language_model() -> None:
    from robot_voice_assistant.motions import is_stop_request

    for said in ["Stop.", "stop walking", "Halt!", "Please stand still", "Okay R1, stop there", "freeze"]:
        assert is_stop_request(said), said
    # stopping by mistake only makes the robot stand; still, these do not ask it to
    for said in ["Keep walking.", "Don't stop", "do not stop walking", "What is a bus stop?", "Turn around"]:
        assert not is_stop_request(said), said


def test_walking_backward_gets_time_for_the_slow_backward_walk() -> None:
    """The walking policy was not trained to walk backward: in the visitor test it went back at about 0.04 m/s
    for 0.3 m/s commanded, and stopped at 0.24 to 0.35 m when its time ran out."""
    controller = ideal()
    pose, t = Pose2D(0.0, 0.0, 0.0), 0.0
    controller.start("walk_backward", pose, 0.0)
    while t < 30.0 and not controller.finished:
        command = controller.update(pose, t)
        pose = Pose2D(pose.x + 0.04 / 0.3 * command.forward * 0.05, 0.0, 0.0)  # the robot does 0.04 of 0.3 m/s
        t += 0.05
    assert pose.x == pytest.approx(-0.5, abs=0.05)


def test_a_turn_on_the_spot_holds_its_place_against_drift() -> None:
    """The walking policy drifts about 0.3 m backward over a turn-around (visitor tests, 2026-11-11), which once
    brought the robot within 0.9 m of a visitor: the controller now walks against the drift while it turns."""
    controller = ideal()
    pose, t, dt = Pose2D(0.0, 0.0, 0.0), 0.0, 0.05
    controller.start("turn_around", pose, 0.0)
    while t < 30.0 and not controller.finished:
        command = controller.update(pose, t)
        forward = command.forward - 0.1  # the policy slips back 0.1 m/s while turning
        pose = Pose2D(pose.x + forward * math.cos(pose.yaw) * dt, pose.y + forward * math.sin(pose.yaw) * dt,
                      pose.yaw + command.turn * dt)
        t += dt
    assert controller.finished
    assert math.hypot(pose.x, pose.y) < 0.1  # without holding its place it would be about 0.6 m away


def test_an_arabic_stop_request_is_recognised_without_the_language_model() -> None:
    from robot_voice_assistant.motions import is_stop_request

    for said in ["توقف!", "قف", "يا روبوت، توقف", "توقف عن المشي", "أوقف"]:
        assert is_stop_request(said), said
    for said in ["لا تتوقف", "استمر في المشي", "أين موقف السيارات؟"]:  # don't stop; keep walking; where is the car park?
        assert not is_stop_request(said), said
