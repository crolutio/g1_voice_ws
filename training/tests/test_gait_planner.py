"""The footstep planner: where each foot should be, relative to the hips, at each moment of the step cycle."""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from foot_placement.gait_planner import GaitPlan, plan_feet  # noqa: E402

PLAN = GaitPlan(period_s=0.8, stance_fraction=0.6, step_height_m=0.08, width_m=0.276, foot_x_m=0.0,
                foot_z_m=-0.62, stand_foot_z_m=-0.64, heel_tilt_rad=0.17)


def feet(command: tuple[float, float, float], phase: float):
    return plan_feet(PLAN, torch.tensor([command]), torch.tensor([phase]))


def test_standing_keeps_both_feet_down_flat_and_shoulder_width_apart() -> None:
    for phase in (0.0, 0.3, 0.7):
        plan = feet((0.0, 0.0, 0.0), phase)
        assert plan.stance.all()
        assert plan.position[0, :, 2].tolist() == pytest.approx([PLAN.stand_foot_z_m] * 2)  # standing straight
        assert plan.position[0, 0, 1] - plan.position[0, 1, 1] == pytest.approx(PLAN.width_m)
        assert plan.position[0, :, 0].tolist() == pytest.approx([0.0, 0.0])
        assert torch.allclose(plan.rotation[0], torch.eye(3).expand(2, 3, 3), atol=1e-6)


def test_walking_forward_the_feet_take_turns_in_front() -> None:
    step = 0.5 * PLAN.period_s * PLAN.stance_fraction  # distance the hips travel while a foot is down at 0.5 m/s
    left_lands = feet((0.5, 0.0, 0.0), 0.0)  # the left foot has just landed
    assert left_lands.position[0, 0, 0] == pytest.approx(step / 2)  # left in front of the hips
    assert left_lands.position[0, 1, 0] < 0  # right behind
    right_lands = feet((0.5, 0.0, 0.0), 0.5)
    assert right_lands.position[0, 1, 0] == pytest.approx(step / 2)  # now the right is in front
    assert right_lands.position[0, 0, 0] < 0


def test_walking_backward_the_landing_foot_is_behind() -> None:
    left_lands = feet((-0.3, 0.0, 0.0), 0.0)
    assert left_lands.position[0, 0, 0] < 0 < left_lands.position[0, 1, 0]


def test_one_foot_is_always_on_the_ground() -> None:
    for phase in torch.linspace(0, 0.99, 100):
        assert feet((0.8, 0.0, 0.3), float(phase)).stance.any()


def test_the_swinging_foot_lifts_most_in_mid_swing() -> None:
    mid_swing = PLAN.stance_fraction + (1 - PLAN.stance_fraction) / 2  # of the left foot's cycle
    plan = feet((0.5, 0.0, 0.0), mid_swing)
    assert not plan.stance[0, 0]
    assert plan.position[0, 0, 2] == pytest.approx(PLAN.foot_z_m + PLAN.step_height_m)


def test_the_swinging_foot_lands_heel_first() -> None:
    just_before_landing = 0.995  # of the left foot's cycle
    rotation = feet((0.5, 0.0, 0.0), just_before_landing).rotation[0, 0]
    toe_up = rotation[2, 0]  # z part of the foot's forward axis
    assert toe_up > math.sin(PLAN.heel_tilt_rad) * 0.8


def test_turning_left_the_planted_feet_turn_with_the_ground() -> None:
    plan = feet((0.0, 0.0, 0.8), 0.3)  # the left foot is planted, half way through its stance
    left_yaw = torch.atan2(plan.rotation[0, 0, 1, 0], plan.rotation[0, 0, 0, 0])
    assert abs(left_yaw.item()) < 1e-3  # mid-stance: straight
    late = feet((0.0, 0.0, 0.8), 0.55)
    assert torch.atan2(late.rotation[0, 0, 1, 0], late.rotation[0, 0, 0, 0]).item() < -0.05  # body turned past it


def test_many_robots_at_once() -> None:
    plan = plan_feet(PLAN, torch.zeros(4096, 3), torch.rand(4096))
    assert plan.position.shape == (4096, 2, 3) and plan.rotation.shape == (4096, 2, 3, 3)
    assert plan.stance.shape == (4096, 2)


def test_before_walking_starts_the_robot_stands_straight_with_feet_shoulder_width_apart() -> None:
    walk_in = torch.tensor([0.0])
    for phase in (0.0, 0.3, 0.7, 0.9):
        plan = plan_feet(PLAN, torch.tensor([[0.5, 0.0, 0.0]]), torch.tensor([phase]), walk_in)
        assert plan.stance.all()
        assert plan.position[0, :, 0].tolist() == pytest.approx([0.0, 0.0])  # side by side
        assert plan.position[0, 0, 1] - plan.position[0, 1, 1] == pytest.approx(PLAN.width_m)
        assert plan.position[0, :, 2].tolist() == pytest.approx([PLAN.stand_foot_z_m] * 2)  # standing straight


def test_walking_in_grows_the_steps_smoothly() -> None:
    lengths = []
    for amount in (0.0, 0.25, 0.5, 1.0):
        plan = plan_feet(PLAN, torch.tensor([[0.5, 0.0, 0.0]]), torch.tensor([0.0]), torch.tensor([amount]))
        lengths.append((plan.position[0, 0, 0] - plan.position[0, 1, 0]).item())
    assert lengths[0] == pytest.approx(0.0) and lengths == sorted(lengths)
    full = plan_feet(PLAN, torch.tensor([[0.5, 0.0, 0.0]]), torch.tensor([0.0]))
    assert lengths[-1] == pytest.approx((full.position[0, 0, 0] - full.position[0, 1, 0]).item())


# --- the foot planter: planted feet stay locked to the ground (like a game engine's foot planting) ---


def walk(planter, command, seconds, dt=0.02, stand_s=0.0, walk_in_s=0.8):
    """Run the planter from standing; returns per step (hips xy, hips yaw, plan, world feet xy)."""
    from foot_placement.gait_planner import FootPlanter  # noqa: F401

    hips_xy, hips_yaw, out = torch.zeros(1, 2), torch.zeros(1), []
    cmd = torch.tensor([command])
    for step in range(int(seconds / dt)):
        walking = step * dt - stand_s
        amount = 0.0 if walking < 0 or not any(command) else 0.5 - 0.5 * math.cos(math.pi * min(walking / walk_in_s, 1.0))
        phase = torch.tensor([(PLAN.stance_fraction + max(walking, 0.0) / PLAN.period_s) % 1.0])
        plan = planter.step(hips_xy, hips_yaw, cmd, phase, torch.tensor([amount]))
        out.append((hips_xy.clone(), hips_yaw.clone(), plan, planter.feet_world_xy.clone()))
        c, s = torch.cos(hips_yaw), torch.sin(hips_yaw)
        velocity = torch.stack([c * cmd[:, 0] - s * cmd[:, 1], s * cmd[:, 0] + c * cmd[:, 1]], 1)
        hips_xy, hips_yaw = hips_xy + dt * amount * velocity, hips_yaw + dt * amount * cmd[:, 2]
    return out


def test_planted_feet_never_slide_even_while_starting_to_walk() -> None:
    from foot_placement.gait_planner import FootPlanter

    for command in ((0.5, 0.0, 0.0), (-0.3, 0.0, 0.0), (0.0, 0.0, 0.8), (0.4, 0.0, 0.4)):
        steps = walk(FootPlanter(PLAN, count=1), command, seconds=3.0, stand_s=0.5)
        for foot in range(2):
            for (_, _, before, xy0), (_, _, after, xy1) in zip(steps, steps[1:]):
                if before.stance[0, foot] and after.stance[0, foot]:
                    assert torch.linalg.norm(xy1[0, foot] - xy0[0, foot]) < 1e-6, command


def test_the_planter_starts_from_standing_straight_with_feet_shoulder_width_apart() -> None:
    from foot_placement.gait_planner import FootPlanter

    first = walk(FootPlanter(PLAN, count=1), (0.5, 0.0, 0.0), seconds=0.4, stand_s=1.0)[-1][2]
    assert first.stance.all()
    assert first.position[0, :, 0].tolist() == pytest.approx([0.0, 0.0], abs=1e-5)
    assert (first.position[0, 0, 1] - first.position[0, 1, 1]).item() == pytest.approx(PLAN.width_m)
    assert first.position[0, :, 2].tolist() == pytest.approx([PLAN.stand_foot_z_m] * 2)


def test_once_walking_each_foot_lands_in_front_of_the_other() -> None:
    from foot_placement.gait_planner import FootPlanter

    steps = walk(FootPlanter(PLAN, count=1), (0.5, 0.0, 0.0), seconds=4.0, walk_in_s=0.8)
    landings = []
    for (_, _, before, _), (_, _, after, _) in zip(steps, steps[1:]):
        for foot in range(2):
            if not before.stance[0, foot] and after.stance[0, foot] and after.position[0, foot, 0] != 0:
                other = 1 - foot
                landings.append((foot, (after.position[0, foot, 0] - after.position[0, other, 0]).item()))
    late = landings[2:]  # after walking in
    assert len(late) >= 4
    assert all(ahead > 0.1 for _, ahead in late)  # the landing foot is in front of the other one
    assert [foot for foot, _ in late[:4]] in ([0, 1, 0, 1], [1, 0, 1, 0])  # left, right, left, right


# --- walking faster: steps get longer up to a limit, then the robot steps more often ---


def test_slow_walking_keeps_the_normal_step_timing() -> None:
    from foot_placement.gait_planner import cycle_period

    assert cycle_period(PLAN, torch.tensor([[0.3, 0.0, 0.0]])).item() == pytest.approx(PLAN.period_s)


def test_fast_walking_steps_more_often_and_caps_the_step_length() -> None:
    from foot_placement.gait_planner import cycle_period

    period = cycle_period(PLAN, torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))
    assert torch.all(period < PLAN.period_s)
    step = 1.0 * period * PLAN.stance_fraction
    assert torch.all(step <= PLAN.max_step_m + 1e-6)
    assert torch.all(period >= PLAN.min_period_s)


# --- each arm swings with its own leg (right leg forward, right arm forward), the elbow bending as it comes forward ---


def arms(command: tuple[float, float, float], phase: float, walk_in: float = 1.0):
    from foot_placement.gait_planner import arm_motion

    return arm_motion(PLAN, torch.tensor([phase]), torch.tensor([command]), torch.tensor([walk_in]))


def test_each_shoulder_swings_with_its_own_leg_and_clearly() -> None:
    shoulder, _ = arms((0.5, 0.0, 0.0), 0.5)  # the right foot has just landed in front, the left is behind
    left, right = shoulder[0].tolist()
    assert right < -0.6 * PLAN.arm_swing_rad  # right arm well forward (negative shoulder pitch is forward on the R1)
    assert left > 0.2  # left arm back
    shoulder, _ = arms((0.5, 0.0, 0.0), 0.0)  # now the left foot is in front
    assert shoulder[0, 0] < -0.6 * PLAN.arm_swing_rad and shoulder[0, 1] > 0.2


def test_the_elbow_bends_as_its_arm_comes_forward() -> None:
    _, elbow = arms((0.5, 0.0, 0.0), 0.5)
    left, right = elbow[0].tolist()
    assert right < -0.6 * PLAN.elbow_swing_rad  # the forward (right) arm's elbow bends more (smaller elbow angle on the R1)
    assert right < left <= 0.0


def test_walking_backward_the_arm_follows_its_leg_back() -> None:
    shoulder, _ = arms((-0.4, 0.0, 0.0), 0.5)  # the right foot has just landed behind
    assert shoulder[0, 1] > 0.2  # right arm back with it


def test_no_arm_motion_when_standing_some_when_turning() -> None:
    shoulder, elbow = arms((0.0, 0.0, 0.0), 0.5, walk_in=0.0)
    assert torch.all(shoulder == 0) and torch.all(elbow == 0)
    turning, _ = arms((0.0, 0.0, 0.8), 0.5)
    assert 0.1 < abs(turning[0, 1].item()) < PLAN.arm_swing_rad


def test_the_arms_grow_in_with_the_walk() -> None:
    half, _ = arms((0.8, 0.0, 0.0), 0.5, walk_in=0.5)
    full, _ = arms((0.8, 0.0, 0.0), 0.5, walk_in=1.0)
    assert abs(half[0, 1].item()) == pytest.approx(abs(full[0, 1].item()) / 2)


# --- gestures from the voice assistant: the walking swing rides on the commanded arm pose, and fades while an arm gestures ---


def test_the_swing_rides_on_a_resting_arm_pose() -> None:
    from foot_placement.gait_planner import swing_on_arm_pose

    rest = torch.tensor([[0.15, 0.15, 1.3, 1.3]])  # shoulders (left, right), then elbows
    swing = torch.tensor([[-0.3, 0.3, -0.2, 0.0]])
    assert torch.allclose(swing_on_arm_pose(rest, rest, swing), rest + swing)


def test_a_gesturing_arm_does_not_swing_but_the_other_arm_does() -> None:
    from foot_placement.gait_planner import swing_on_arm_pose

    rest = torch.tensor([[0.15, 0.15, 1.3, 1.3]])
    waving = torch.tensor([[0.15, -2.5, 1.3, 0.4]])  # right arm raised for a wave
    swing = torch.tensor([[-0.3, 0.3, -0.2, 0.0]])
    targets = swing_on_arm_pose(waving, rest, swing)
    assert torch.allclose(targets[0, [1, 3]], waving[0, [1, 3]])  # the waving arm follows the gesture exactly
    assert torch.allclose(targets[0, [0, 2]], (rest + swing)[0, [0, 2]])  # the left arm keeps swinging


# --- weight shift: the plan puts the hips over the standing foot, and pulls them back over the feet ---


def test_the_hips_shift_over_the_standing_foot_while_walking() -> None:
    from foot_placement.gait_planner import FootPlanter

    steps = walk(FootPlanter(PLAN, count=1), (0.5, 0.0, 0.0), seconds=4.0, walk_in_s=0.8)[100:]  # walking fully
    on_left = [p.hips_offset[0, 1].item() for _, _, p, _ in steps if p.stance[0].tolist() == [True, False]]
    on_right = [p.hips_offset[0, 1].item() for _, _, p, _ in steps if p.stance[0].tolist() == [False, True]]
    assert on_left and on_right
    assert max(on_left) > 0.8 * PLAN.hip_sway_m  # over the left foot (left is +y)
    assert min(on_right) < -0.8 * PLAN.hip_sway_m
    assert max(abs(v) for v in on_left + on_right) < PLAN.width_m / 2  # never past the standing foot


def test_no_sway_when_standing() -> None:
    from foot_placement.gait_planner import FootPlanter

    steps = walk(FootPlanter(PLAN, count=1), (0.0, 0.0, 0.0), seconds=1.0)
    assert all(torch.allclose(p.hips_offset, torch.zeros(1, 2), atol=1e-6) for _, _, p, _ in steps)


def test_hips_that_drift_off_the_feet_are_pulled_back_over_them() -> None:
    from foot_placement.gait_planner import FootPlanter

    planter = FootPlanter(PLAN, count=1)
    stand = (torch.zeros(1, 3), torch.tensor([PLAN.stance_fraction]), torch.zeros(1))
    planter.step(torch.zeros(1, 2), torch.zeros(1), *stand)  # feet placed under the hips
    drifted = planter.step(torch.tensor([[0.0, -0.05]]), torch.zeros(1), *stand)  # hips 5 cm to the right
    assert drifted.hips_offset[0].tolist() == pytest.approx([0.0, 0.05], abs=1e-6)
    # the leg targets are from where the hips should be, so the feet keep their normal place under them
    assert drifted.position[0, :, 1].tolist() == pytest.approx([PLAN.width_m / 2, -PLAN.width_m / 2], abs=1e-6)


# --- stopping eases out over a step, like starting eases in (no snapping to standing with a foot in the air) ---


def test_walk_in_rises_while_moving_and_falls_gradually_after_a_stop() -> None:
    from foot_placement.gait_planner import step_walk_in

    walk_in = torch.tensor([1.0])
    after_stop = step_walk_in(walk_in, torch.tensor([False]), dt=0.02, walk_in_s=0.8)
    assert after_stop.item() == pytest.approx(1.0 - 0.02 / 0.8)  # not straight to 0
    rising = step_walk_in(torch.tensor([0.0]), torch.tensor([True]), dt=0.02, walk_in_s=0.8)
    assert rising.item() == pytest.approx(0.02 / 0.8)
    assert step_walk_in(torch.tensor([0.01]), torch.tensor([False]), dt=0.02, walk_in_s=0.8).item() == 0.0


def test_while_stopping_the_robot_finishes_its_step_and_brings_the_foot_under_the_hips() -> None:
    from foot_placement.gait_planner import FootPlanter

    planter = FootPlanter(PLAN, count=1)
    hips = (torch.zeros(1, 2), torch.zeros(1))
    stopped = torch.zeros(1, 3)
    planter.step(*hips, stopped, torch.tensor([PLAN.stance_fraction]), torch.zeros(1))  # standing
    mid_swing = planter.step(*hips, stopped, torch.tensor([0.8]), torch.tensor([0.5]))  # the left foot swings
    assert not mid_swing.stance[0, 0]  # still stepping, not forced down
    assert mid_swing.position[0, 0, 2] > PLAN.stand_foot_z_m  # lifted
    landed = planter.step(*hips, stopped, torch.tensor([0.0]), torch.tensor([0.4]))
    assert landed.stance[0, 0]
    assert planter.feet_world_xy[0, 0].tolist() == pytest.approx([0.0, PLAN.width_m / 2], abs=1e-3)  # under the hips


# --- catching itself: like a person, the next step goes where the body is tipping (capture point foot placement) ---


def _landing_after(velocity_xy: tuple[float, float]) -> torch.Tensor:
    from foot_placement.gait_planner import FootPlanter

    planter = FootPlanter(PLAN, count=1)
    hips, command = (torch.zeros(1, 2), torch.zeros(1)), torch.tensor([[0.3, 0.0, 0.0]])
    planter.step(*hips, command, torch.tensor([PLAN.stance_fraction]), torch.zeros(1))
    planter.step(*hips, command, torch.tensor([0.99]), torch.ones(1), hips_velocity=torch.tensor([velocity_xy]))
    return planter.feet_world_xy[0, 0]  # the left foot, at the end of its swing


def test_a_body_moving_as_commanded_steps_normally() -> None:
    from foot_placement.gait_planner import FootPlanter

    as_commanded = _landing_after((0.3, 0.0))
    planter = FootPlanter(PLAN, count=1)
    hips, command = (torch.zeros(1, 2), torch.zeros(1)), torch.tensor([[0.3, 0.0, 0.0]])
    planter.step(*hips, command, torch.tensor([PLAN.stance_fraction]), torch.zeros(1))
    planter.step(*hips, command, torch.tensor([0.99]), torch.ones(1))  # no velocity given: no correction
    assert torch.allclose(as_commanded, planter.feet_world_xy[0, 0], atol=1e-5)


def test_tipping_sideways_or_forward_moves_the_next_step_that_way() -> None:
    normal = _landing_after((0.3, 0.0))
    tipping_left = _landing_after((0.3, 0.2))
    rushing = _landing_after((0.6, 0.0))
    gain = math.sqrt(0.7 / 9.81)
    assert (tipping_left - normal)[1].item() == pytest.approx(gain * 0.2, rel=0.05)
    assert (rushing - normal)[0].item() == pytest.approx(gain * 0.3, rel=0.05)
    falling_fast = _landing_after((0.3, 2.0))
    assert (falling_fast - normal)[1].item() == pytest.approx(PLAN.max_catch_step_m, rel=0.01)  # capped


# --- reactive stepping: a pushed or stumbling robot starts stepping quickly, even when told to stand ---


def test_a_robot_moving_off_its_command_needs_to_step() -> None:
    from foot_placement.gait_planner import needs_to_step

    command = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    velocity = torch.tensor([[0.05, 0.0], [0.5, 0.05], [0.0, 0.35]])  # still; walking as told; pushed sideways
    assert needs_to_step(PLAN, command, velocity).tolist() == [False, False, True]


def test_stumbling_starts_the_steps_much_faster_than_a_calm_start() -> None:
    from foot_placement.gait_planner import step_walk_in

    calm = step_walk_in(torch.zeros(1), torch.tensor([True]), dt=0.02, walk_in_s=0.8)
    stumbling = step_walk_in(torch.zeros(1), torch.tensor([True]), dt=0.02, walk_in_s=0.8,
                             stumbling=torch.tensor([True]))
    assert stumbling.item() >= 3 * calm.item()


# --- arm commands move smoothly: a gesture can never make an arm jump ---


def test_arm_targets_move_toward_the_goal_at_a_limited_speed() -> None:
    from foot_placement.gait_planner import move_toward

    current = torch.tensor([[0.15, -1.5]])
    goal = torch.tensor([[0.15, 0.15]])
    moved = move_toward(current, goal, max_step=0.06)
    assert moved[0].tolist() == pytest.approx([0.15, -1.44])
    assert move_toward(torch.tensor([[0.10]]), torch.tensor([[0.12]]), max_step=0.06).item() == pytest.approx(0.12)


# --- the policy sees the joints without the gesture: a raised arm looks like a resting (swinging) arm ---


def test_gesture_offsets_are_taken_out_of_what_the_policy_sees() -> None:
    from foot_placement.gait_planner import without_gesture

    joint_pos = torch.tensor([[0.5, -1.35, 0.2]])  # a leg joint, a raised shoulder, a turned head
    joint_vel = torch.tensor([[1.0, 2.0, 0.5]])
    offset = torch.tensor([[0.0, -1.5, 0.2]])  # the gesture moved the shoulder by -1.5 and the head by 0.2
    offset_rate = torch.tensor([[0.0, 2.0, 0.5]])
    pos, vel = without_gesture(joint_pos, joint_vel, offset, offset_rate)
    assert pos[0].tolist() == pytest.approx([0.5, 0.15, 0.0], abs=1e-6)
    assert vel[0].tolist() == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)


# --- centre of mass: when the arms move the body's weight, the hips shift the other way to keep it over the feet ---


def test_a_centre_of_mass_shift_moves_the_hips_the_other_way() -> None:
    from foot_placement.gait_planner import FootPlanter

    planter = FootPlanter(PLAN, count=1)
    stand = (torch.zeros(1, 3), torch.tensor([PLAN.stance_fraction]), torch.zeros(1))
    planter.step(torch.zeros(1, 2), torch.zeros(1), *stand)
    shifted = planter.step(torch.zeros(1, 2), torch.zeros(1), *stand, com_shift=torch.tensor([[0.03, -0.01]]))
    assert shifted.hips_offset[0].tolist() == pytest.approx([-0.03, 0.01], abs=1e-6)
    # the feet targets are from where the hips should be: 3 cm further forward of them, 1 cm further right
    assert shifted.position[0, :, 0].tolist() == pytest.approx([0.03, 0.03], abs=1e-6)


def test_the_centre_of_mass_correction_is_limited() -> None:
    from foot_placement.gait_planner import FootPlanter

    planter = FootPlanter(PLAN, count=1)
    stand = (torch.zeros(1, 3), torch.tensor([PLAN.stance_fraction]), torch.zeros(1))
    planter.step(torch.zeros(1, 2), torch.zeros(1), *stand)
    shifted = planter.step(torch.zeros(1, 2), torch.zeros(1), *stand, com_shift=torch.tensor([[0.5, 0.0]]))
    assert shifted.hips_offset[0, 0].item() == pytest.approx(-PLAN.max_com_shift_m)
