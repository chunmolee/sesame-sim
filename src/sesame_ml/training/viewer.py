"""Shows the live state of PPO standing-training environments in one window.

Every robot pose comes from a training worker. This module only renders
kinematics; it runs neither a second policy nor a second physics simulation.
Environments are placed at fixed offsets on one shared floor. The offsets are
for display only and do not mean the environments collide with each other.
"""

from __future__ import annotations

import math
import time
from typing import Any

import mujoco
import numpy as np

from sesame_ml.model import mesh_directory, mjcf_path

_ARENA_XML = """
<mujoco model="sesame_live_training">
  <visual>
    <global azimuth="135" elevation="-45"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7"/>
  </visual>
  <asset>
    <texture name="ground_tex" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.18 0.20 0.22" rgb2="0.28 0.30 0.32"/>
    <material name="ground_mat" texture="ground_tex" texrepeat="16 16"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="ground_mat"/>
  </worldbody>
</mujoco>
"""


def build_training_scene(count: int) -> tuple[mujoco.MjModel, np.ndarray, np.ndarray]:
    """Builds a shared floor with several robots for display.

    Args:
        count: Number of training environments to show. Must be a positive
            integer.

    Returns:
        A tuple ``(model, root_addresses, offsets)``. Each robot's ``qpos``
        keeps the layout of the source model. ``offsets`` are display-only
        horizontal displacements in meters.
    """
    if not isinstance(count, int) or count <= 0:
        raise ValueError("number of displayed environments must be a positive integer")
    assets = {path.name: path.read_bytes() for path in mesh_directory().glob("*.stl")}
    robot = mujoco.MjSpec.from_string(mjcf_path().read_text(encoding="utf-8"), assets=assets)
    arena = mujoco.MjSpec.from_string(_ARENA_XML)
    # The attached bodies reference the original STL file names, so the
    # composed model only needs one copy of the mesh bytes.
    arena.assets.update(assets)
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    offsets = np.zeros((count, 3))
    for index in range(count):
        offsets[index, :2] = [
            (index % columns - (columns - 1) / 2) * 0.28,
            ((rows - 1) / 2 - index // columns) * 0.28,
        ]
        # attach_body moves the source body. Copy the spec for each attach so
        # it does not consume the source needed by later environments.
        child = robot.copy()
        frame = arena.worldbody.add_frame()
        frame.attach_body(child.body("base"), prefix=f"env{index}/")
    model = arena.compile()
    addresses = np.asarray(
        [int(model.joint(f"env{i}/root").qposadr[0]) for i in range(count)], dtype=np.int32
    )
    return model, addresses, offsets


class TrainingViewer:
    """Multi-environment live training view in a single GLFW window.

    The window is created in the main process; workers need no graphics
    context. ready() throttles snapshot RPCs and rendering, but does not limit
    PPO sampling speed. When the window is closed, update() returns False so
    the training callback stops sampling and the existing model-saving flow is
    reused.
    """

    def __init__(self, number_of_environments: int, frames_per_second: float = 20.0):
        """Creates the window and rendering resources for all environments.

        Args:
            number_of_environments: Number of training environments to show.
            frames_per_second: Maximum display rate used by ready().

        Raises:
            ValueError: If the display rate or environment count is invalid.
            RuntimeError: If GLFW or the OpenGL window cannot be created.
        """
        import glfw

        if not math.isfinite(frames_per_second) or frames_per_second <= 0:
            raise ValueError("visualization FPS must be finite and positive")
        self._glfw = glfw
        self._window = None
        self._context = None
        self._initialized = False
        self._interval = 1 / frames_per_second
        self._next_frame = 0.0
        self._count = number_of_environments
        self._model, self._addresses, self._offsets = build_training_scene(number_of_environments)
        self._data = mujoco.MjData(self._model)
        self._qpos_size = self._model.nq // number_of_environments
        self._scene = mujoco.MjvScene(
            self._model, maxgeom=self._model.ngeom + number_of_environments + 32
        )
        self._option = mujoco.MjvOption()
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self._camera)
        self._camera.lookat[:] = [0, 0, 0.04]
        self._camera.distance = max(0.55, math.ceil(math.sqrt(self._count)) * 0.46)
        self._camera.azimuth = 135
        self._camera.elevation = -45
        self._mouse_position = (0.0, 0.0)
        try:
            if not glfw.init():
                raise RuntimeError("Live training requires a working desktop display (GLFW)")
            self._initialized = True
            self._window = glfw.create_window(
                1100, 800, "Sesame PPO - live parallel standing", None, None
            )
            if self._window is None:
                raise RuntimeError("Could not create the live training OpenGL window")
            glfw.make_context_current(self._window)
            # Leave vertical sync off so normal training is never implicitly
            # throttled by the monitor refresh rate.
            glfw.swap_interval(0)
            self._context = mujoco.MjrContext(self._model, mujoco.mjtFontScale.mjFONTSCALE_150)
            glfw.set_cursor_pos_callback(self._window, self._move_mouse)
            glfw.set_mouse_button_callback(self._window, self._mouse_button)
            glfw.set_scroll_callback(self._window, self._scroll)
            glfw.set_key_callback(self._window, self._key)
        except BaseException:
            self.close()
            raise

    def ready(self) -> bool:
        """Returns whether the next display time has been reached.

        A close event also makes this return True so the callback handles it
        immediately.
        """
        if self._window is None:
            return True
        self._glfw.poll_events()
        return self._glfw.window_should_close(self._window) or time.monotonic() >= self._next_frame

    def update(self, poses: list[np.ndarray], *, samples_seen: int) -> bool:
        """Updates the view from real worker qpos snapshots.

        Args:
            poses: One raw MuJoCo ``qpos`` per training environment, not a
                normalized observation.
            samples_seen: Cumulative number of samples PPO has collected. Used
                only for the on-screen label.

        Returns:
            False if the window is closed, otherwise True.

        Raises:
            ValueError: If the snapshot count or contents are invalid.
        """
        glfw = self._glfw
        if self._window is None or glfw.window_should_close(self._window):
            return False
        if len(poses) != self._count:
            raise ValueError("live view snapshot count does not match training environments")
        for index, pose in enumerate(poses):
            pose = np.asarray(pose, dtype=np.float64)
            if pose.shape != (self._qpos_size,) or not np.isfinite(pose).all():
                raise ValueError("live view requires finite raw MuJoCo qpos snapshots")
            start = self._addresses[index]
            self._data.qpos[start : start + self._qpos_size] = pose
            self._data.qpos[start : start + 3] += self._offsets[index]
        # The composed scene never calls mj_step. Doing so would show a
        # physics trajectory different from the training workers'.
        mujoco.mj_forward(self._model, self._data)
        mujoco.mjv_updateScene(
            self._model,
            self._data,
            self._option,
            None,
            self._camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self._scene,
        )
        for index in range(self._count):
            label = self._scene.geoms[self._scene.ngeom]
            position = self._data.qpos[self._addresses[index] : self._addresses[index] + 3].copy()
            position[2] += 0.1
            mujoco.mjv_initGeom(
                label,
                mujoco.mjtGeom.mjGEOM_LABEL,
                np.zeros(3),
                position,
                np.eye(3).ravel(),
                np.ones(4, dtype=np.float32),
            )
            label.label = f"Env {index + 1}"
            self._scene.ngeom += 1
        glfw.make_context_current(self._window)
        width, height = glfw.get_framebuffer_size(self._window)
        if width > 0 and height > 0:
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjr_render(viewport, self._scene, self._context)
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                f"LIVE PPO | {self._count} training environments\nSamples: {samples_seen:,}",
                "",
                self._context,
            )
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                viewport,
                "Actual training states | drag: rotate | wheel: zoom | Esc: save and stop",
                "",
                self._context,
            )
            glfw.swap_buffers(self._window)
        glfw.poll_events()
        self._next_frame = time.monotonic() + self._interval
        return not glfw.window_should_close(self._window)

    def _mouse_button(self, window: Any, button: int, action: int, mods: int) -> None:
        self._mouse_position = self._glfw.get_cursor_pos(window)

    def _move_mouse(self, window: Any, x: float, y: float) -> None:
        glfw = self._glfw
        dx, dy = x - self._mouse_position[0], y - self._mouse_position[1]
        self._mouse_position = (x, y)
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) != glfw.PRESS:
            return
        _, height = glfw.get_window_size(window)
        if height > 0:
            mujoco.mjv_moveCamera(
                self._model,
                mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                dx / height,
                dy / height,
                self._scene,
                self._camera,
            )

    def _scroll(self, window: Any, x: float, y: float) -> None:
        mujoco.mjv_moveCamera(
            self._model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * y, self._scene, self._camera
        )

    def _key(self, window: Any, key: int, scancode: int, action: int, mods: int) -> None:
        if key == self._glfw.KEY_ESCAPE and action == self._glfw.PRESS:
            self._glfw.set_window_should_close(window, True)

    def close(self) -> None:
        """Frees rendering resources before destroying the window."""
        if self._window is not None:
            self._glfw.make_context_current(self._window)
            if self._context is not None:
                self._context.free()
                self._context = None
            self._glfw.destroy_window(self._window)
            self._window = None
        if self._initialized:
            self._glfw.terminate()
            self._initialized = False
