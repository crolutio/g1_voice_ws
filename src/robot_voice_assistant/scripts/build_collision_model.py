"""Build config/robot_collision_model.npz from Unitree's R1 URDF and meshes.

The URDF and meshes come from https://github.com/unitreerobotics/unitree_ros/tree/master/robots/r1_description
(a copy is in ~/g1_voice_ws/robots/r1). Run:

    python build_collision_model.py ~/g1_voice_ws/robots/r1/R1.urdf ~/g1_voice_ws/robots/r1
"""

from __future__ import annotations

import json
import os
import struct
import sys
import xml.etree.ElementTree as ElementTree

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robot_voice_assistant.kinematics import ARM_LINKS, BODY_LINKS, DEFAULT_MODEL_PATH, origin_transform  # noqa: E402

POINTS_PER_LINK = 150


def read_binary_stl(path: str) -> np.ndarray:
    with open(path, "rb") as file:
        file.seek(80)
        (count,) = struct.unpack("<I", file.read(4))
        data = np.frombuffer(file.read(count * 50), dtype=np.uint8).reshape(count, 50)
    triangles = data[:, 12:48].copy().view("<f4").reshape(count, 3, 3)
    return triangles.reshape(-1, 3).astype(np.float64)


def sample_points(vertices: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    """Evenly spread surface points: farthest point sampling on the (deduplicated) mesh vertices."""
    unique = np.unique(np.round(vertices, 5), axis=0)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(unique)))]
    distances = np.linalg.norm(unique - unique[chosen[0]], axis=1)
    for _ in range(min(count, len(unique)) - 1):
        chosen.append(int(np.argmax(distances)))
        distances = np.minimum(distances, np.linalg.norm(unique - unique[chosen[-1]], axis=1))
    return unique[chosen]


def main() -> None:
    urdf_path, mesh_dir = sys.argv[1], sys.argv[2]
    robot = ElementTree.parse(urdf_path).getroot()
    joints = []
    for joint in robot.findall("joint"):
        origin = joint.find("origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0.0] * 3
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0.0] * 3
        axis_element = joint.find("axis")
        axis = [float(v) for v in axis_element.get("xyz").split()] if axis_element is not None else [0.0] * 3
        joints.append(
            {
                "name": joint.get("name"),
                "parent": joint.find("parent").get("link"),
                "child": joint.find("child").get("link"),
                "origin": origin_transform(xyz, rpy).tolist(),
                "axis": axis if joint.get("type") != "fixed" else [0.0, 0.0, 0.0],
            }
        )
    arrays = {"joints": np.array(json.dumps(joints))}
    meshes = {
        link.get("name"): link.find("visual/geometry/mesh").get("filename")
        for link in robot.findall("link")
        if link.find("visual/geometry/mesh") is not None
    }
    for link in (*BODY_LINKS, *ARM_LINKS):
        vertices = read_binary_stl(os.path.join(mesh_dir, meshes[link]))
        arrays[f"points/{link}"] = sample_points(vertices, POINTS_PER_LINK)
        print(f"{link}: {len(vertices)} vertices -> {len(arrays[f'points/{link}'])} points")
    np.savez_compressed(os.path.abspath(DEFAULT_MODEL_PATH), **arrays)
    print(f"wrote {os.path.abspath(DEFAULT_MODEL_PATH)}")


if __name__ == "__main__":
    main()
