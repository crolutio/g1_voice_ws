"""The Newton viewer window in its own process, so drawing never slows the simulation down.

The simulation (walk_scene.py) runs physics, the policy and the walk on its main thread; drawing the 3D window took
about a third of each step there. Here the window lives in a separate process, on its own CPU core:

- at the start it gets the robot (the URDF), the camera, and every still mesh (the hall, booths, products), and
  rarely-changing meshes later (the product on the stand), through a queue;
- while running it reads the robot's link poses and each moving mesh (visitors, the object in a visitor's hand)
  from shared memory, which the simulation overwrites at the viewer's rate. Nothing waits for anything: the window
  draws the latest poses it finds, as fast as it can (up to ``max_fps``).

    RemoteViewer(...)        # in the simulation: starts the process, then .log_mesh(...) / .publish_robot(...)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from multiprocessing import shared_memory

import numpy as np

URDF = os.path.expanduser("~/g1_voice_ws/robots/r1/R1.urdf")
# the same lens as Isaac Lab's Newton viewer: 12 mm focal length on USD's default 15.2908 mm vertical aperture
CAMERA_FOV_DEG = float(np.degrees(2.0 * np.arctan(15.2908 / (2.0 * 12.0))))


# --- shared memory blocks: one per moving thing, a sequence number first so the reader knows something changed ---

class _Slot:
    """A float32 array in shared memory, after an int64 sequence number (bumped after every write)."""

    def __init__(self, shape: tuple[int, ...], name: str | None = None) -> None:
        size = 8 + int(np.prod(shape)) * 4
        self.shm = shared_memory.SharedMemory(name=name, create=name is None, size=size)
        self.seq = np.ndarray((1,), dtype=np.int64, buffer=self.shm.buf, offset=0)
        self.data = np.ndarray(shape, dtype=np.float32, buffer=self.shm.buf, offset=8)
        if name is None:
            self.seq[0] = 0

    def write(self, values: np.ndarray) -> None:
        self.data[...] = values
        self.seq[0] += 1

    def close(self, unlink: bool = False) -> None:
        self.shm.close()
        if unlink:
            self.shm.unlink()


# --- the simulation's side ---

class RemoteViewer:
    """Drop-in for the Newton visualizer that GitexHall draws into, with the window in another process."""

    accepts_numpy = True  # GitexHall passes numpy arrays straight here (no GPU round trip)
    _viewer = None  # GitexHall looks for the in-process window; there is none here

    def __init__(self, body_names: list[str], eye, lookat, width: int = 1600, height: int = 900,
                 max_fps: float = 60.0) -> None:
        context = mp.get_context("spawn")  # a fresh interpreter: no Isaac Sim / Kit state is inherited
        self._queue = context.Queue()
        self._robot = _Slot((len(body_names), 7))
        self._slots: dict[str, _Slot] = {}
        self._process = context.Process(
            target=_run, name="newton-viewer", daemon=True,
            args=(self._queue, list(body_names), self._robot.shm.name, tuple(eye), tuple(lookat), width, height,
                  max_fps),
        )
        # a spawned process normally re-imports the parent's main script (walk_scene.py, which imports Isaac Lab):
        # hide it for the start, so the viewer process only imports this module
        import sys

        main = sys.modules["__main__"]
        hidden = {key: getattr(main, key) for key in ("__file__", "__spec__") if hasattr(main, key)}
        for key in hidden:
            setattr(main, key, None) if key == "__spec__" else delattr(main, key)
        try:
            self._process.start()
        finally:
            for key, value in hidden.items():
                setattr(main, key, value)

    def alive(self) -> bool:
        return self._process.is_alive()

    def publish_robot(self, body_poses: np.ndarray) -> None:
        """(bodies, 7) link poses in the world: x, y, z, qx, qy, qz, qw (Isaac Lab's body order)."""
        self._robot.write(body_poses)

    def log_mesh(self, name: str, points: np.ndarray, indices: np.ndarray, colour, roughness: float = 0.7,
                 uvs: np.ndarray | None = None, texture: str | None = None, dynamic: bool = False,
                 normals: np.ndarray | None = None) -> None:
        points = np.ascontiguousarray(points, dtype=np.float32)
        if not dynamic:  # still or rarely changing: send the whole mesh once
            self._queue.put(("mesh", name, points, np.asarray(indices, dtype=np.int32), tuple(colour),
                             float(roughness), None if uvs is None else np.asarray(uvs, np.float32), texture, None))
            return
        slot = self._slots.get(name)
        width = 6 if normals is not None else 3
        if slot is None or slot.data.shape != (len(points), width):  # a new moving mesh: its own shared block
            slot = self._slots[name] = _Slot((len(points), width))
            self._queue.put(("dynamic", name, slot.shm.name, len(points), width,
                             np.asarray(indices, dtype=np.int32), tuple(colour), float(roughness)))
        values = points if normals is None else np.concatenate([points, normals], axis=1)
        slot.write(values)

    def close(self) -> None:
        try:
            self._queue.put(("quit",))
            self._process.join(timeout=5)
        except Exception:
            pass
        if self._process.is_alive():
            self._process.terminate()
        for slot in [self._robot, *self._slots.values()]:
            slot.close(unlink=True)


# --- the viewer process ---

def _pitch_yaw(eye, lookat) -> tuple[float, float]:
    """Newton's camera angles (degrees) looking from ``eye`` at ``lookat`` (z up)."""
    dx, dy, dz = (lookat[i] - eye[i] for i in range(3))
    yaw = np.degrees(np.arctan2(dy, dx))
    pitch = np.degrees(np.arctan2(dz, np.hypot(dx, dy)))
    return float(pitch), float(yaw)


def _body_map(model, body_names: list[str]) -> np.ndarray:
    """For each Newton body, the index of the same link in the simulation's body list (-1 if it has none)."""
    labels = list(getattr(model, "body_label", None) or getattr(model, "body_key", []))
    index = {name: i for i, name in enumerate(body_names)}
    mapping = np.full(len(labels), -1, dtype=np.int64)
    for i, label in enumerate(labels):
        mapping[i] = index.get(str(label).split("/")[-1], -1)
    return mapping


def _run(queue, body_names, robot_shm, eye, lookat, width, height, max_fps) -> None:
    import queue as queue_module

    import warp as wp

    wp.config.quiet = True
    import newton
    import newton.viewer

    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    builder = newton.ModelBuilder()
    builder.add_urdf(URDF, floating=True, collapse_fixed_joints=True)  # fixed links merged, as in Isaac Lab
    model = builder.finalize(device=device)
    state = model.state()
    mapping = _body_map(model, body_names)
    found = mapping >= 0
    print(f"[viewer] {int(found.sum())} of {len(mapping)} robot links matched to the simulation", flush=True)
    body_q = state.body_q.numpy()

    viewer = newton.viewer.ViewerGL(width=width, height=height, vsync=False)
    viewer.set_model(model)
    viewer.show_ui = False
    pitch, yaw = _pitch_yaw(eye, lookat)
    viewer.set_camera(wp.vec3(*eye), pitch, yaw)
    viewer.camera.fov = CAMERA_FOV_DEG

    robot = _Slot((len(body_names), 7), name=robot_shm)
    moving: dict[str, tuple] = {}  # name: (slot, last seq, indices on the device, width)
    textured: list[str] = []
    last_robot = -1
    frame_time = 1.0 / max_fps

    def enable_texture(name: str) -> None:
        mesh = viewer.objects.get(viewer._qualify(name)) if hasattr(viewer, "_qualify") else viewer.objects.get(name)
        if mesh is not None and mesh.material[3] != 1.0:
            r, m, c, _ = mesh.material
            mesh.material = (r, m, c, 1.0)

    def log(name, points, indices, colour, roughness, uvs=None, texture=None, normals=None, dynamic=False):
        viewer.log_mesh(
            name,
            wp.array(points, dtype=wp.vec3, device=device),
            indices if isinstance(indices, wp.array) else wp.array(indices, dtype=wp.int32, device=device),
            normals=None if normals is None else wp.array(np.ascontiguousarray(normals), dtype=wp.vec3, device=device),
            uvs=None if uvs is None else wp.array(uvs, dtype=wp.vec2, device=device),
            texture=texture, color=colour, roughness=roughness, dynamic=dynamic, backface_culling=False,
        )

    started = time.monotonic()
    while viewer.is_running():
        frame_start = time.monotonic()
        while True:  # meshes and new moving meshes from the simulation
            try:
                message = queue.get_nowait()
            except queue_module.Empty:
                break
            if message[0] == "quit":
                viewer.close()
                return
            if message[0] == "mesh":
                _, name, points, indices, colour, roughness, uvs, texture, _ = message
                log(name, points, indices, colour, roughness, uvs, texture)
                if texture is not None:
                    textured.append(name)
            elif message[0] == "dynamic":
                _, name, shm_name, count, width_, indices, colour, roughness = message
                if name in moving:
                    moving[name][0].close()
                moving[name] = (_Slot((count, width_), name=shm_name), -1,
                                wp.array(indices, dtype=wp.int32, device=device), width_, colour, roughness)

        viewer.begin_frame(time.monotonic() - started)
        seq = int(robot.seq[0])
        if seq != last_robot:
            poses = robot.data.copy()
            body_q[found] = poses[mapping[found]]
            state.body_q.assign(body_q)
            last_robot = seq
        viewer.log_state(state)
        for name, (slot, last, indices, width_, colour, roughness) in list(moving.items()):
            seq = int(slot.seq[0])
            if seq == last:
                continue
            values = slot.data.copy()
            log(name, values[:, :3], indices, colour, roughness,
                normals=values[:, 3:6] if width_ == 6 else None, dynamic=True)
            moving[name] = (slot, seq, indices, width_, colour, roughness)
        for name in textured:
            enable_texture(name)
        textured.clear()
        viewer.end_frame()
        spare = frame_time - (time.monotonic() - frame_start)
        if spare > 0:
            time.sleep(spare)
