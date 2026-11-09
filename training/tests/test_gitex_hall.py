"""The GITEX hall's layout: valid meshes, carpets that do not flicker, visitors that keep clear of the robot."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gitex_hall import hall_parts, visitor_routes  # noqa: E402
from visitors import Visitor  # noqa: E402


def test_every_part_of_the_hall_is_a_valid_mesh() -> None:
    parts = hall_parts(0.0, 0.0)
    assert len(parts) > 100
    for part in parts:
        assert part.points.ndim == 2 and part.points.shape[1] == 3, part.name
        assert len(part.indices) % 3 == 0 and part.indices.max() < len(part.points), part.name
        assert all(0.0 <= c <= 1.0 for c in part.colour), part.name


def test_carpets_sit_clear_of_the_ground_and_of_each_other() -> None:
    parts = {p.name: p for p in hall_parts(0.0, 0.0)}
    hall_top = parts["carpet"].points[:, 2].max()
    assert parts["carpet"].points[:, 2].min() >= 0.0 and hall_top >= 0.01  # clear of the physics ground (z = 0)
    booth_carpets = [p for name, p in parts.items() if name.endswith("_carpet")]
    assert booth_carpets
    for carpet in booth_carpets:
        assert carpet.points[:, 2].max() - hall_top >= 0.01, carpet.name  # a gap the depth buffer can tell apart


def test_visitors_keep_clear_of_the_robot_except_where_they_stop_to_look() -> None:
    for route in visitor_routes(0.0, 0.0):
        visitor = Visitor(route, speed=1.0, cycle_m=2.8, frames=80, standing_frame=40)
        stops = [(w.x, w.y) for w in route if w.pause_s > 0]
        for step in np.arange(0.0, visitor.loop_s, 0.1):
            x, y, _, _ = visitor.pose(float(step))
            near_a_stop = any(math.dist((x, y), stop) < 1.0 for stop in stops)
            assert math.hypot(x, y) >= 1.2 or near_a_stop, (route, x, y)


def test_visitors_are_split_into_one_mesh_per_clothing_colour() -> None:
    from gitex_hall import visitor_colour_groups

    # two visitors of 4 points each, two triangles each: one in region 0 (shoes), one in region 2 (shirt)
    indices = np.array([0, 1, 2, 2, 3, 1], dtype=np.int32)
    regions = np.array([0, 0, 2, 2], dtype=np.int8)
    black, white, red = (0.1, 0.1, 0.1), (0.9, 0.9, 0.9), (0.8, 0.1, 0.1)
    outfits = [(black, black, white, white), (black, black, red, red)]
    groups = visitor_colour_groups(indices, regions, outfits, points_each=4)
    colours = [colour for colour, _, _ in groups]
    assert sorted(colours) == sorted([black, white, red])  # the shoes of both visitors share one mesh
    triangles = 0
    for colour, vertex_ids, local in groups:
        assert local.max() < len(vertex_ids)
        triangles += len(local) // 3
        if colour == red:  # the second visitor's shirt: its points are offset by one visitor
            assert set(vertex_ids.tolist()) == {5, 6, 7}
    assert triangles == 4  # every triangle of every visitor drawn exactly once
