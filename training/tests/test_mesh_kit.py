"""Triangle meshes for the GITEX hall: the same triangles go to the Isaac Sim camera and to the Newton viewer."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mesh_kit import MeshPart, box, cylinder, merge_by_colour, sphere, triangulate  # noqa: E402


def test_a_box_has_flat_faces_and_its_size() -> None:
    part = box("crate", center=(1.0, 2.0, 0.5), size=(2.0, 1.0, 1.0), colour=(0.5, 0.2, 0.1))
    assert part.points.shape == (24, 3)  # 4 corners per face, so every face is flat-shaded
    assert part.indices.shape == (36,)  # 2 triangles per face
    assert part.points.min(axis=0).tolist() == pytest.approx([0.0, 1.5, 0.0])
    assert part.points.max(axis=0).tolist() == pytest.approx([2.0, 2.5, 1.0])


def test_box_faces_point_outward() -> None:
    part = box("crate", center=(0, 0, 0), size=(1, 1, 1), colour=(1, 1, 1))
    triangles = part.points[part.indices.reshape(-1, 3)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    centres = triangles.mean(axis=1)
    assert np.all(np.einsum("ij,ij->i", normals, centres) > 0)  # counter-clockwise seen from outside


def test_a_turned_box_turns_about_its_centre() -> None:
    part = box("wall", center=(0, 0, 1), size=(4, 0.2, 2), colour=(1, 1, 1), yaw=math.pi / 2)
    extent = part.points.max(axis=0) - part.points.min(axis=0)
    assert extent.tolist() == pytest.approx([0.2, 4.0, 2.0])


def test_cylinder_and_sphere_sizes() -> None:
    pot = cylinder("pot", base=(0, 0, 0), radius=0.3, height=0.5, colour=(1, 1, 1), segments=32)
    assert pot.points[:, 2].min() == pytest.approx(0.0) and pot.points[:, 2].max() == pytest.approx(0.5)
    assert np.linalg.norm(pot.points[:, :2], axis=1).max() == pytest.approx(0.3, abs=1e-6)
    ball = sphere("bush", center=(0, 0, 1), radius=0.5, colour=(0, 1, 0), scale=(1, 1, 0.8))
    assert ball.points[:, 2].max() == pytest.approx(1.4, abs=1e-6)
    assert ball.indices.max() < len(ball.points)


def test_parts_of_one_colour_are_merged_into_one_mesh() -> None:
    red = (0.8, 0.1, 0.1)
    parts = [box("a", (0, 0, 0), (1, 1, 1), red), box("b", (3, 0, 0), (1, 1, 1), red), box("c", (6, 0, 0), (1, 1, 1), (0, 0, 1))]
    merged = merge_by_colour(parts, prefix="hall")
    assert len(merged) == 2
    reds = next(m for m in merged if m.colour == red)
    assert reds.points.shape == (48, 3) and reds.indices.shape == (72,)
    assert reds.indices.max() == 47  # the second box's indices are offset past the first box's points


def test_polygons_are_split_into_triangles() -> None:
    indices = triangulate(np.array([4, 3]), np.array([0, 1, 2, 3, 4, 5, 6]))
    assert indices.tolist() == [0, 1, 2, 0, 2, 3, 4, 5, 6]


def test_a_mesh_part_keeps_its_texture_coordinates() -> None:
    part = MeshPart("skin", np.zeros((3, 3), np.float32), np.array([0, 1, 2], np.int32), (1, 1, 1),
                    uvs=np.zeros((3, 2), np.float32), texture="skin.png")
    assert part.uvs.shape == (3, 2) and part.texture == "skin.png"
