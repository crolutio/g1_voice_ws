"""Walking people for the GITEX hall: NVIDIA's biped character, skinned for every frame of its walk cycle.

NVIDIA's dressed characters use a different skeleton than the walk animation (Reallusion, 101 joints, against the
animation's 81-joint biped), so the walk never reaches them; the biped character has exactly the animation's
skeleton. Its mesh is skinned here (linear blend skinning) for each of the 80 walk frames once, and cached, so the
scene only has to pick a frame: the same points go to the robot's camera (USD) and to the Newton viewer.

Frames are in the character's own frame: metres, z up, facing +x, feet on z = 0.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np

ASSETS = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/People"
CHARACTER_URL = f"{ASSETS}/Characters/biped_demo/biped_demo_meters.usd"
WALK_URL = f"{ASSETS}/Animations/stand_walk_loop_in_place.skelanim.usd"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "people")
FRAMES = 80  # the walk loop: time codes 0 to 80 at 30 per second, frame 80 is frame 0 again
FPS = 30.0
# body regions, by height on the standing character (metres) and distance from the middle (the hands)
SHOES, TROUSERS, SHIRT, SKIN = 0, 1, 2, 3
REGIONS = 4


@dataclass
class WalkCycle:
    frames: np.ndarray  # (FRAMES, N, 3) float32 points
    indices: np.ndarray  # (3M,) int32 triangles
    regions: np.ndarray  # (N,) int8 body region of each point
    cycle_m: float  # ground covered by one walk cycle (metres)
    standing_frame: int  # the frame with the feet closest together

    @property
    def walk_speed(self) -> float:
        """m/s at which the feet keep pace with the ground (one cycle per FRAMES / FPS seconds)."""
        return self.cycle_m / (FRAMES / FPS)


def _download(url: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, os.path.basename(url))
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def load_walk_cycle() -> WalkCycle:
    """The cached walk cycle, computed on first use (needs USD: run in Isaac Sim's Python)."""
    cache = os.path.join(CACHE_DIR, "walk_cycle.npz")
    if os.path.exists(cache):
        data = np.load(cache)
        return WalkCycle(data["frames"], data["indices"], data["regions"], float(data["cycle_m"]),
                         int(data["standing_frame"]))
    cycle = _compute_walk_cycle(_download(CHARACTER_URL), _download(WALK_URL))
    np.savez_compressed(cache, frames=cycle.frames, indices=cycle.indices, regions=cycle.regions,
                        cycle_m=cycle.cycle_m, standing_frame=cycle.standing_frame)
    return cycle


def _compute_walk_cycle(character_path: str, walk_path: str) -> WalkCycle:
    from pxr import Usd, UsdGeom, UsdSkel

    from mesh_kit import triangulate

    stage = Usd.Stage.CreateInMemory()
    stage.SetTimeCodesPerSecond(FPS)  # as the animation's layer: otherwise its times are rescaled
    stage.DefinePrim("/Person").GetReferences().AddReference(character_path)
    walk = stage.DefinePrim("/Walk")
    walk.GetReferences().AddReference(walk_path)
    skeleton = UsdSkel.Skeleton(stage.GetPrimAtPath("/Person/Root"))
    UsdSkel.BindingAPI.Apply(skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets([walk.GetPath()])
    root = UsdSkel.Root(stage.GetPrimAtPath("/Person"))
    cache = UsdSkel.Cache()
    cache.Populate(root, Usd.TraverseInstanceProxies())
    skeleton_query = cache.GetSkelQuery(skeleton)
    binding = cache.ComputeSkelBindings(root, Usd.TraverseInstanceProxies())[0]
    skinning = binding.GetSkinningTargets()[0]
    mesh = UsdGeom.Mesh(skinning.GetPrim())

    rest = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    bind = np.array(skinning.GetGeomBindTransform(), dtype=np.float64)
    rest = np.c_[rest, np.ones(len(rest))] @ bind  # USD points are row vectors
    joint_indices, joint_weights = skinning.ComputeJointInfluences()
    per_point = skinning.GetNumInfluencesPerComponent()
    joint_indices = np.array(joint_indices, dtype=np.int64).reshape(-1, per_point)
    joint_weights = np.array(joint_weights, dtype=np.float64).reshape(-1, per_point)
    order = [str(j).split("/")[-1] for j in skeleton_query.GetJointOrder()]
    # the mesh may list the joints in its own order (skel:joints): its influences index that list
    mesh_joints = UsdSkel.BindingAPI(mesh.GetPrim()).GetJointsAttr().Get()
    full_order = [str(j) for j in skeleton_query.GetJointOrder()]
    to_skeleton = np.array([full_order.index(str(j)) for j in mesh_joints] if mesh_joints else range(len(order)))
    left, right = order.index("L_Ankle"), order.index("R_Ankle")
    frames, ankles = [], []
    for frame in range(FRAMES):
        transforms = np.array(skeleton_query.ComputeSkinningTransforms(frame), dtype=np.float64)[to_skeleton]  # (J, 4, 4)
        skinned = np.zeros((len(rest), 3))
        for k in range(per_point):  # blend: sum of weight * (point @ joint transform)
            skinned += joint_weights[:, k:k + 1] * np.einsum("pi,pij->pj", rest, transforms[joint_indices[:, k]])[:, :3]
        frames.append(skinned)
        joints = skeleton_query.ComputeJointSkelTransforms(frame)
        ankles.append([np.array(joints[left].ExtractTranslation()), np.array(joints[right].ExtractTranslation())])

    # y up, facing +z (USD character)  ->  z up, facing +x (the scene): (x, y, z) -> (z, x, y)
    frames = np.stack(frames)[:, :, [2, 0, 1]].astype(np.float32)
    frames[:, :, 2] -= frames[0, :, 2].min()  # feet on the ground
    ankles = np.array(ankles)  # (FRAMES, 2, 3) in the character's frame (y up, +z forward)

    # ground covered per cycle: the planted (lower) foot slides backward at the walking speed in an in-place walk
    lower = np.argmin(ankles[:, :, 1], axis=1)
    forward = ankles[np.arange(FRAMES), lower, 2]
    step = -(np.roll(forward, -1) - forward)  # backward motion of the planted foot per frame
    same_foot = lower == np.roll(lower, -1)
    cycle_m = float(np.median(step[same_foot]) * FRAMES)
    standing_frame = int(np.argmin(np.abs(ankles[:, 0, 2] - ankles[:, 1, 2])))

    standing = frames[standing_frame]
    regions = np.full(len(standing), SHIRT, dtype=np.int8)
    height = standing[:, 2]
    regions[height < 0.95] = TROUSERS
    regions[height < 0.09] = SHOES
    regions[height > 1.5] = SKIN  # neck and head
    hands = (height > 0.6) & (height < 0.95) & (np.abs(standing[:, 1]) > 0.22)
    regions[hands] = SKIN
    indices = triangulate(np.array(mesh.GetFaceVertexCountsAttr().Get()), np.array(mesh.GetFaceVertexIndicesAttr().Get()))
    return WalkCycle(frames, indices, regions, cycle_m, standing_frame)


def outfit_texture(shoes, trousers, shirt, skin) -> np.ndarray:
    """A small texture with one colour per body region (columns), for the region texture coordinates."""
    colours = np.array([shoes, trousers, shirt, skin], dtype=np.float32)
    return (np.repeat(np.repeat(colours[None, :, :], 4, axis=0), 16, axis=1) * 255).astype(np.uint8)  # (4, 64, 3)


def region_uvs(regions: np.ndarray) -> np.ndarray:
    """Texture coordinates that pick each point's region column in outfit_texture()."""
    return np.stack([(regions.astype(np.float32) + 0.5) / REGIONS, np.full(len(regions), 0.5, np.float32)], axis=1)
