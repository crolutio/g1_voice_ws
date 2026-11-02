"""Preview the planned walk without physics: footstep plan -> IK -> animated stick figure, plus checks.

The hips move along a straight line (level, upper body upright); the feet follow gait_planner.py and the legs
come from leg_kinematics.py. Prints how well the IK hits the plan, whether planted feet stay put on the ground,
the knee angles, step length and foot spacing, and writes one GIF per command.

    ~/g1_voice_ws/.venv/bin/python -m foot_placement.preview --out-dir /tmp/preview
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from foot_placement.gait_planner import FootPlanter, GaitPlan, cycle_period
from foot_placement.leg_kinematics import LegKinematics, rotation_error

URDF = os.path.expanduser("~/g1_voice_ws/robots/r1/R1.urdf")
COMMANDS = {"forward 1.0 m/s": (1.0, 0.0, 0.0), "forward 0.5 m/s": (0.5, 0.0, 0.0), "backward 0.5 m/s": (-0.5, 0.0, 0.0),
            "turn left 1.0 rad/s": (0.0, 0.0, 1.0)}
PLAN = GaitPlan()
STAND_S, WALK_IN_S = 1.0, 0.8  # stand straight first, then grow into the walk over one step cycle
DT = 0.02
TORSO_M, HEAD_M = 0.40, 0.12  # upright upper body drawn above the hips
PANEL, SCALE = 380, 330


def simulate(command: tuple[float, float, float], seconds: float, legs: dict[str, LegKinematics]):
    """Joint angles, joint points and foot plans over time, with the hips on a straight (or turning) line."""
    steps = int(seconds / DT)
    cmd = torch.tensor([command])
    angles = {side: legs[side].default_angles() for side in legs}
    hips_xy, hips_yaw = np.zeros(2), 0.0  # the straight (or turning) line the hips travel along
    shift = 0.0  # the hips' sideways shift from that line (the weight shift), as if they follow the plan exactly
    planter = FootPlanter(PLAN, count=1)
    period = float(cycle_period(PLAN, cmd)[0])
    cycle = PLAN.stance_fraction  # the left foot takes the first step: its swing starts at phase = stance_fraction
    frames = []
    for step in range(steps):
        walking_s = step * DT - STAND_S
        phase = torch.tensor([cycle % 1.0])
        if walking_s >= 0:
            cycle += DT / period
        amount = 0.0 if walking_s < 0 else 0.5 - 0.5 * math.cos(math.pi * min(walking_s / WALK_IN_S, 1.0))
        left = np.array([-math.sin(hips_yaw), math.cos(hips_yaw)])
        actual = hips_xy + shift * left
        plan = planter.step(torch.tensor([actual], dtype=torch.float32), torch.tensor([hips_yaw], dtype=torch.float32),
                            cmd, phase, torch.tensor([amount]))
        shift += float(plan.hips_offset[0, 1])
        frame = {"hips_xy": hips_xy + shift * left, "hips_yaw": hips_yaw, "stance": plan.stance[0].numpy(),
                 "sway": shift,
                 "hips_z": -(PLAN.stand_foot_z_m + (PLAN.foot_z_m - PLAN.stand_foot_z_m) * amount),
                 "status": "standing straight" if walking_s < 0 else ("starting to walk" if amount < 1 else "walking")}
        for index, side in enumerate(("left", "right")):
            target_position, target_rotation = plan.position[:, index], plan.rotation[:, index]
            angles[side] = legs[side].solve(target_position, target_rotation, angles[side], iterations=15)
            position, rotation, _, points = legs[side]._forward(angles[side])
            frame[side] = {
                "angles": angles[side][0].numpy().copy(),
                "points": torch.cat([points[0], position], 0).numpy(),  # 6 joints + ankle, hips frame
                "error_m": float(torch.linalg.norm(position - target_position)),
                "error_deg": math.degrees(float(torch.linalg.norm(rotation_error(rotation, target_rotation)))),
                "target": target_position[0].numpy(),
            }
        frames.append(frame)
        c, s = math.cos(hips_yaw), math.sin(hips_yaw)
        hips_xy = hips_xy + DT * amount * np.array([c * command[0] - s * command[1], s * command[0] + c * command[1]])
        hips_yaw += DT * amount * command[2]
    return frames


def to_world(frame: dict, point: np.ndarray) -> np.ndarray:
    c, s = math.cos(frame["hips_yaw"]), math.sin(frame["hips_yaw"])
    return np.array([frame["hips_xy"][0] + c * point[0] - s * point[1], frame["hips_xy"][1] + s * point[0] + c * point[1], point[2]])


def checks(name: str, frames: list[dict]) -> None:
    errors_m = [f[side]["error_m"] for f in frames for side in ("left", "right")]
    errors_deg = [f[side]["error_deg"] for f in frames for side in ("left", "right")]
    knees = np.degrees([f[side]["angles"][3] for f in frames for side in ("left", "right")])
    slip = 0.0  # how far a planted foot moves on the ground while planted
    for index, side in enumerate(("left", "right")):
        planted = None
        for f in frames:
            here = to_world(f, f[side]["points"][-1])[:2]
            if f["stance"][index]:
                planted = here if planted is None else planted
                slip = max(slip, float(np.linalg.norm(here - planted)))
            else:
                planted = None
    width = np.mean([abs(f["left"]["points"][-1][1] - f["right"]["points"][-1][1]) for f in frames])
    both_down = [f for f in frames if f["stance"].all()]
    length = np.mean([abs(f["left"]["points"][-1][0] - f["right"]["points"][-1][0]) for f in both_down]) if both_down else 0
    sway = max(abs(f["sway"]) for f in frames)
    print(f"{name:22s} hips sway {sway * 100:3.1f} cm; IK error max {max(errors_m) * 1000:5.1f} mm / {max(errors_deg):4.1f} deg; planted-foot slip "
          f"{slip * 100:4.1f} cm; knee {knees.min():3.0f} to {knees.max():3.0f} deg; feet {width * 100:4.1f} cm apart; "
          f"step {length * 100:4.1f} cm")


def draw(frames: list[dict], name: str, path: str) -> None:
    images = []
    colours = {"left": (40, 110, 230), "right": (230, 130, 30)}
    for step, f in enumerate(frames[::2]):
        image = Image.new("RGB", (2 * PANEL + 60, PANEL + 70), "white")
        d = ImageDraw.Draw(image)
        d.text((20, 10), f"planned walk (IK, no physics)  |  {name}  |  t = {step * 2 * DT:4.1f} s  |  {f['status']}",
               fill=(0, 0, 0))
        d.text((20, 28), "feet: green = planted, red = swinging   legs: blue = left, orange = right   upper body upright",
               fill=(90, 90, 90))
        for panel, (axis, title) in enumerate(((0, "from the side (walking direction ->)"), (1, "from the front"))):
            ox, oy = 20 + panel * (PANEL + 20), 50
            ground = oy + PANEL - 30
            d.rectangle([ox, oy, ox + PANEL, oy + PANEL], outline=(200, 200, 200))
            d.text((ox + 8, oy + 6), title, fill=(0, 0, 0))
            d.line([ox, ground, ox + PANEL, ground], fill=(120, 120, 120), width=2)
            travelled = f["hips_xy"][0] if axis == 0 else 0.0
            for k in range(-6, 8):
                x = ox + PANEL / 2 + (k * 0.25 - travelled % 0.25) * SCALE
                if axis == 0 and ox <= x <= ox + PANEL:
                    d.line([x, ground, x - 10, ground + 12], fill=(160, 160, 160), width=2)
            hips_height = f["hips_z"] + 0.07  # ankles are about 7 cm above the sole

            def px(p: np.ndarray) -> tuple[float, float]:
                h = p[0] if axis == 0 else -p[1]
                return ox + PANEL / 2 + h * SCALE, ground - (hips_height + p[2]) * SCALE

            hips = np.zeros(3)
            d.line([px(hips), px(np.array([0, 0, TORSO_M]))], fill=(60, 60, 60), width=7)
            head = px(np.array([0, 0, TORSO_M + HEAD_M]))
            d.ellipse([head[0] - 14, head[1] - 14, head[0] + 14, head[1] + 14], outline=(60, 60, 60), width=3)
            for index, side in enumerate(("left", "right")):
                points = f[side]["points"]
                chain = [hips, points[0], points[3], points[5]]  # hips, hip joint, knee, ankle
                for a, b in zip(chain, chain[1:]):
                    d.line([px(a), px(b)], fill=colours[side], width=6)
                knee = px(points[3])
                d.ellipse([knee[0] - 5, knee[1] - 5, knee[0] + 5, knee[1] + 5], fill=colours[side])
                ankle = px(points[5])
                d.ellipse([ankle[0] - 8, ankle[1] - 8, ankle[0] + 8, ankle[1] + 8],
                          fill=(40, 170, 60) if f["stance"][index] else (220, 40, 40))
        images.append(image)
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(2 * DT * 1000 * 2), loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", default="preview")
    parser.add_argument("--seconds", type=float, default=4.4)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    legs = {side: LegKinematics.from_urdf(URDF, side) for side in ("left", "right")}
    for name, command in COMMANDS.items():
        frames = simulate(command, args.seconds, legs)
        checks(name, frames)
        draw(frames, name, os.path.join(args.out_dir, name.replace(" ", "_").replace("/", "") + ".gif"))
    print(f"GIFs (half speed) in {args.out_dir}")


if __name__ == "__main__":
    main()
