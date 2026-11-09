"""Plain triangle meshes (numpy) for the GITEX hall.

Every piece of the hall is built here as triangles, then written both into the USD stage (for the robot's camera)
and into the Newton viewer (log_mesh), so the two always show the same thing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

Colour = tuple[float, float, float]


@dataclass
class MeshPart:
    name: str
    points: np.ndarray  # (N, 3) float32, world metres (z up)
    indices: np.ndarray  # (3M,) int32, counter-clockwise seen from outside
    colour: Colour
    uvs: np.ndarray | None = None  # (N, 2) float32, with ``texture``
    texture: str | None = None  # image path
    roughness: float = 0.7
    tags: dict = field(default_factory=dict)


def _turned(points: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    turned = points.copy()
    turned[:, 0] = c * points[:, 0] - s * points[:, 1]
    turned[:, 1] = s * points[:, 0] + c * points[:, 1]
    return turned


def box(name: str, center: Sequence[float], size: Sequence[float], colour: Colour, yaw: float = 0.0,
        roughness: float = 0.7) -> MeshPart:
    """A box with flat-shaded faces (4 corners each), turned by ``yaw`` about its centre."""
    hx, hy, hz = (s / 2 for s in size)
    faces = [  # each face: 4 corners counter-clockwise seen from outside
        [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)],  # +x
        [(-hx, hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz)],  # -x
        [(hx, hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz)],  # +y
        [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)],  # -y
        [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)],  # +z
        [(-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz), (-hx, -hy, -hz)],  # -z
    ]
    points = _turned(np.array([corner for face in faces for corner in face], dtype=np.float32), yaw)
    points += np.asarray(center, dtype=np.float32)
    indices = np.array([base + i for base in range(0, 24, 4) for i in (0, 1, 2, 0, 2, 3)], dtype=np.int32)
    return MeshPart(name, points, indices, colour, roughness=roughness)


def cylinder(name: str, base: Sequence[float], radius: float, height: float, colour: Colour,
             segments: int = 24, roughness: float = 0.7) -> MeshPart:
    """An upright cylinder standing on ``base``, with top and bottom caps."""
    angles = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    bottom = np.column_stack([ring, np.zeros(segments)])
    top = np.column_stack([ring, np.full(segments, height)])
    centres = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, height]])
    points = np.vstack([bottom, top, centres]).astype(np.float32) + np.asarray(base, dtype=np.float32)
    bottom_centre, top_centre = 2 * segments, 2 * segments + 1
    indices = []
    for i in range(segments):
        j = (i + 1) % segments
        indices += [i, j, segments + j, i, segments + j, segments + i]  # side
        indices += [top_centre, segments + i, segments + j]  # top cap
        indices += [bottom_centre, j, i]  # bottom cap
    return MeshPart(name, points, np.array(indices, dtype=np.int32), colour, roughness=roughness)


def sphere(name: str, center: Sequence[float], radius: float, colour: Colour, scale: Sequence[float] = (1, 1, 1),
           rings: int = 12, segments: int = 18, roughness: float = 0.7) -> MeshPart:
    """A sphere (or, with ``scale``, an ellipsoid)."""
    points = [(0.0, 0.0, radius)]
    for i in range(1, rings):
        theta = math.pi * i / rings
        for j in range(segments):
            phi = 2 * math.pi * j / segments
            points.append((radius * math.sin(theta) * math.cos(phi), radius * math.sin(theta) * math.sin(phi),
                           radius * math.cos(theta)))
    points.append((0.0, 0.0, -radius))
    array = np.array(points, dtype=np.float32) * np.asarray(scale, dtype=np.float32) + np.asarray(center, dtype=np.float32)
    bottom = len(points) - 1
    indices = []
    for j in range(segments):  # top fan
        indices += [0, 1 + j, 1 + (j + 1) % segments]
    for i in range(rings - 2):
        for j in range(segments):
            a, b = 1 + i * segments + j, 1 + i * segments + (j + 1) % segments
            indices += [a, a + segments, b + segments, a, b + segments, b]
    last = 1 + (rings - 2) * segments
    for j in range(segments):  # bottom fan
        indices += [last + j, bottom, last + (j + 1) % segments]
    return MeshPart(name, array, np.array(indices, dtype=np.int32), colour, roughness=roughness)


def triangulate(face_vertex_counts: np.ndarray, face_vertex_indices: np.ndarray) -> np.ndarray:
    """Fan-triangulate USD-style polygons (counts and indices) into a flat triangle index list."""
    triangles = []
    start = 0
    for count in face_vertex_counts.tolist():
        for k in range(1, count - 1):
            triangles += [face_vertex_indices[start], face_vertex_indices[start + k], face_vertex_indices[start + k + 1]]
        start += count
    return np.array(triangles, dtype=np.int32)


def merge_by_colour(parts: Sequence[MeshPart], prefix: str) -> list[MeshPart]:
    """One mesh per colour (untextured parts), so the viewer draws a few large meshes instead of hundreds."""
    groups: dict[tuple, list[MeshPart]] = {}
    for part in parts:
        groups.setdefault((part.colour, part.roughness), []).append(part)
    merged = []
    for number, ((colour, roughness), members) in enumerate(groups.items()):
        offsets = np.cumsum([0] + [len(m.points) for m in members[:-1]])
        points = np.vstack([m.points for m in members]).astype(np.float32)
        indices = np.concatenate([m.indices + offset for m, offset in zip(members, offsets)]).astype(np.int32)
        merged.append(MeshPart(f"{prefix}_{number}", points, indices, colour, roughness=roughness))
    return merged
