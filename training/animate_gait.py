"""Turn a record_gait.py recording into animated GIFs: a stick figure seen from the side and the front.

Feet are green on the ground and red in the air; "JUMP" shows when both feet are in the air. Ground
stripes move under the robot, so you can see how far it walks. For turning, a top view shows the feet.

    ~/g1_voice_ws/.venv/bin/python animate_gait.py gait_recording.npz --out-dir gifs --label "current policy"
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

SEGMENTS = [
    ("pelvis_link", "waist_yaw_link"), ("waist_yaw_link", "head_yaw_link"),
    ("pelvis_link", "left_hip_pitch_link"), ("left_hip_pitch_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_pitch_link"), ("left_ankle_pitch_link", "left_ankle_roll_link"),
    ("pelvis_link", "right_hip_pitch_link"), ("right_hip_pitch_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_pitch_link"), ("right_ankle_pitch_link", "right_ankle_roll_link"),
    ("waist_yaw_link", "left_shoulder_pitch_link"), ("left_shoulder_pitch_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_roll_link"),
    ("waist_yaw_link", "right_shoulder_pitch_link"), ("right_shoulder_pitch_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_roll_link"),
]
LEFT, RIGHT = (40, 110, 230), (230, 130, 30)  # left leg blue, right leg orange
PANEL, SCALE = 360, 300  # pixels per panel, pixels per metre


def side(name: str) -> tuple[int, int, int]:
    return LEFT if name.startswith("left") else RIGHT if name.startswith("right") else (60, 60, 60)


def draw_panel(draw: ImageDraw.ImageDraw, origin: tuple[int, int], points: dict[str, np.ndarray], axis: int,
               ground_shift: float, contacts: np.ndarray, title: str) -> None:
    """Draw one view. ``axis`` 0 = side view (forward is right), 1 = front view (robot's left is left)."""
    ox, oy = origin
    ground_y = oy + PANEL - 40
    draw.rectangle([ox, oy, ox + PANEL, oy + PANEL], outline=(200, 200, 200))
    draw.text((ox + 8, oy + 6), title, fill=(0, 0, 0))
    draw.line([ox, ground_y, ox + PANEL, ground_y], fill=(120, 120, 120), width=2)
    if axis == 0:  # moving ground stripes every 0.25 m
        for k in range(-4, 6):
            x = ox + PANEL / 2 + (k * 0.25 - ground_shift % 0.25) * SCALE
            if ox <= x <= ox + PANEL:
                draw.line([x, ground_y, x - 10, ground_y + 12], fill=(160, 160, 160), width=2)

    def to_px(p: np.ndarray) -> tuple[float, float]:
        horizontal = p[0] if axis == 0 else -p[1]
        return ox + PANEL / 2 + horizontal * SCALE, ground_y - p[2] * SCALE

    for start, end in SEGMENTS:
        draw.line([to_px(points[start]), to_px(points[end])], fill=side(end), width=5)
    for foot, down in (("left_ankle_roll_link", contacts[0]), ("right_ankle_roll_link", contacts[1])):
        x, y = to_px(points[foot])
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(40, 170, 60) if down else (220, 40, 40))
    for knee in ("left_knee_link", "right_knee_link"):
        x, y = to_px(points[knee])
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=side(knee))


def draw_top(draw: ImageDraw.ImageDraw, origin: tuple[int, int], trail: list[np.ndarray], points: dict[str, np.ndarray],
             contacts: np.ndarray) -> None:
    ox, oy = origin
    draw.rectangle([ox, oy, ox + PANEL, oy + PANEL], outline=(200, 200, 200))
    draw.text((ox + 8, oy + 6), "from above", fill=(0, 0, 0))
    centre = trail[0]

    def to_px(p: np.ndarray) -> tuple[float, float]:
        return ox + PANEL / 2 + (p[0] - centre[0]) * SCALE, oy + PANEL / 2 - (p[1] - centre[1]) * SCALE

    for a, b in zip(trail, trail[1:]):
        draw.line([to_px(a), to_px(b)], fill=(190, 190, 190), width=2)
    pelvis = points["pelvis_link"]
    for foot, down, colour in (("left_ankle_roll_link", contacts[0], LEFT), ("right_ankle_roll_link", contacts[1], RIGHT)):
        x, y = to_px(points[foot])
        draw.line([to_px(pelvis), (x, y)], fill=colour, width=4)
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(40, 170, 60) if down else (220, 40, 40))
    x, y = to_px(pelvis)
    draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(60, 60, 60))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("recording")
    parser.add_argument("--out-dir", default="gifs")
    parser.add_argument("--label", default="")
    parser.add_argument("--slow", type=float, default=1.0, help="2 = half speed")
    args = parser.parse_args()
    data = np.load(args.recording)
    positions, contacts, yaw = data["positions"], data["contacts"], data["yaw"]
    bodies, commands, dt = [str(b) for b in data["bodies"]], [str(c) for c in data["commands"]], float(data["dt"])
    os.makedirs(args.out_dir, exist_ok=True)
    every = 2  # 25 frames per second
    for robot, command in enumerate(commands):
        turning = "turn" in command
        frames = []
        start = positions[robot, 0, bodies.index("pelvis_link"), :2]
        trail = []
        for t in range(0, positions.shape[1], every):
            world = positions[robot, t]
            pelvis = world[bodies.index("pelvis_link")]
            c, s = np.cos(-yaw[robot, t]), np.sin(-yaw[robot, t])
            local = {}
            for i, name in enumerate(bodies):  # the robot's own frame: x forward, y left, z up
                dx, dy = world[i, 0] - pelvis[0], world[i, 1] - pelvis[1]
                local[name] = np.array([c * dx - s * dy, s * dx + c * dy, world[i, 2]])
            travelled = np.dot(pelvis[:2] - start, [np.cos(yaw[robot, 0]), np.sin(yaw[robot, 0])])
            trail.append(pelvis[:2])
            panels = 3 if turning else 2
            image = Image.new("RGB", (PANEL * panels + 20 * (panels + 1), PANEL + 70), "white")
            draw = ImageDraw.Draw(image)
            feet = contacts[robot, t]
            header = f"{args.label}  |  {command}  |  t = {t * dt:4.1f} s"
            draw.text((20, 12), header, fill=(0, 0, 0))
            draw.text((20, 30), "feet: green = on the ground, red = in the air   legs: blue = left, orange = right",
                      fill=(90, 90, 90))
            if not feet.any():
                draw.rectangle([PANEL * panels - 60, 8, PANEL * panels + 40, 40], fill=(220, 40, 40))
                draw.text((PANEL * panels - 40, 17), "JUMP", fill=(255, 255, 255))
            draw_panel(draw, (20, 60), local, 0, travelled, feet, "from the side (walking direction ->)")
            draw_panel(draw, (PANEL + 40, 60), local, 1, 0.0, feet, "from the front")
            if turning:
                draw_top(draw, (2 * PANEL + 60, 60), trail, {n: world[i] for i, n in enumerate(bodies)}, feet)
            frames.append(image)
        name = command.split(" ")[0] + ("_left" if "left" in command else "")
        path = os.path.join(args.out_dir, f"{name}.gif")
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 * dt * every * args.slow), loop=0)
        jumping = float(np.mean(~contacts[robot].any(axis=1)))
        print(f"{path}: {len(frames)} frames, both feet in the air {jumping:.0%} of the time")


if __name__ == "__main__":
    main()
