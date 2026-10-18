"""Native MuJoCo tasks rebuilt from the published Sesame deployment spec.

The upstream repository does not provide this module. The reward weights
follow its MJX locomotion tasks; the task success criteria and the
43-dimensional observation follow docs/architecture.md and simulation.md.
This file is a local reconstruction, not the unpublished upstream source file.

Common conventions:
    * Robot frame: +x forward (OLED/camera direction), +y left, +z up.
    * The 8 joints always use the firmware channel order
      sesame_ml.constants.JOINT_NAMES (R1, R2, L1, L2, R4, R3, L3, L4), in
      radians.
    * Actions are residuals relative to the stand pose STAND_ANGLES_RAD
      (normalized to [-1, 1]; multiplied by action_scale_rad they become
      radians). Servo targets and joint positions are absolute radians.
    * Velocity commands are [forward m/s, lateral m/s, yaw rate rad/s].

Slices of the 43-dimensional proprioceptive observation (all float32):
    * [0:3]   Direction of world +z (opposite to gravity) in the body frame
              (third row of the rotation matrix).
    * [3:6]   IMU acceleration divided by 9.81 (in g).
    * [6:9]   IMU gyroscope angular velocity (rad/s).
    * [9:17]  Servo targets minus the stand pose (radians); in privileged
              mode, true joint positions minus the stand pose instead.
    * [17:25] Privileged mode only: joint velocities divided by 10,
              otherwise 0.
    * [25:33] Previous step's normalized action.
    * [33:36] Velocity command divided by [0.25, 0.25, 1.4].
    * [36:40] Privileged mode only: touch flags of the four feet
              (r3, r4, l3, l4), otherwise 0.
    * [40:43] Privileged mode, navigation task only: target position relative
              to the body in the body frame (meters), otherwise 0.

Privilege boundary: without privileged_joint_state, true joint state, ground
contact and target-relative position are not provided directly. The
navigation command is still computed from the simulator's true pose and goal;
it is a baseline command supplied by the task and does not imply that real
hardware has localization or obstacle avoidance.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from sesame_ml.constants import (
    CONTROL_DT,
    JOINT_LIMITS_RAD,
    PHYSICS_DT,
    SERVO_MAX_SPEED_RAD_S,
    STAND_ANGLES_RAD,
    clip_servo_targets,
    validate_joint_vector,
)
from sesame_ml.model import (
    joint_dof_addresses,
    joint_qpos_addresses,
    load_model,
    make_data,
    set_stand_pose,
)


class Task(StrEnum):
    """Task kind; also determines reset, reward terms and termination.

    Attributes:
        STAND: Hold the stand pose until the step limit; not falling is
            success.
        RECOVERY: Recover from a sideways-fallen initial pose and stand
            stably.
        LOCOMOTION: Track randomly changing velocity/turning commands.
        NAVIGATION: Walk to the green target marker; obstacle contact is
            penalized separately and is not a success criterion.
    """

    STAND = "stand"
    RECOVERY = "recovery"
    LOCOMOTION = "locomotion"
    NAVIGATION = "navigation"

    @property
    def default_episode_steps(self) -> int:
        """Default episode step limit, in control steps (not physics steps)."""

        return {Task.STAND: 500, Task.RECOVERY: 600, Task.LOCOMOTION: 1000, Task.NAVIGATION: 1500}[
            self
        ]


@dataclass(frozen=True)
class SesameEnvConfig:
    """Immutable configuration of SesameEnv.

    Validated by __post_init__ after construction; a string task is converted
    to Task.

    Attributes:
        task: Task kind; either a Task or its string value.
        control_dt: Control period in seconds, 0.02 s (50 Hz) by default;
            must be an integer multiple of physics_dt.
        physics_dt: MuJoCo physics time step in seconds.
        action_scale_rad: Radian offset from the stand pose that an action of
            +-1 maps to.
        domain_randomization: If True, DomainParameters are sampled on every
            reset; otherwise the nominal (default) parameters are used.
        privileged_joint_state: If True, the observation includes true joint
            state, ground contact and target-relative position. Simulation
            only; not available on the deployed robot.
        observation_mode: "state" returns only the 43-dimensional vector;
            "pixels" returns a dict with "proprio" and "rgb".
        maximum_episode_steps: Episode step limit; None uses the task default.
        camera_width: Front camera image width in pixels.
        camera_height: Front camera image height in pixels.
        camera_hz: Front camera update rate in Hz, in simulation time; the
            observation reuses the previous image between updates.
        render_width: Width in pixels of the external rgb_array rendering.
        render_height: Height in pixels of the external rgb_array rendering.
    """

    task: Task | str = Task.STAND
    control_dt: float = CONTROL_DT
    physics_dt: float = PHYSICS_DT
    action_scale_rad: float = 0.58
    domain_randomization: bool = True
    privileged_joint_state: bool = False
    observation_mode: str = "state"
    maximum_episode_steps: int | None = None
    camera_width: int = 160
    camera_height: int = 120
    camera_hz: float = 15.0
    render_width: int = 640
    render_height: int = 480

    def __post_init__(self) -> None:
        """Validates the fields and normalizes task to a Task.

        Raises:
            ValueError: A time/scale/frequency field is not finite and
                positive, control_dt is not an integer multiple of
                physics_dt, observation_mode is invalid, a pixel size is not
                a positive integer, or maximum_episode_steps is not positive.
        """

        # The dataclass is frozen, so write the normalized task back through
        # object.__setattr__.
        object.__setattr__(self, "task", Task(self.task))
        for name in ("control_dt", "physics_dt", "action_scale_rad", "camera_hz"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        ratio = self.control_dt / self.physics_dt
        if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-9):
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        if self.observation_mode not in {"state", "pixels"}:
            raise ValueError("observation_mode must be 'state' or 'pixels'")
        for name in ("camera_width", "camera_height", "render_width", "render_height"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.maximum_episode_steps is not None and self.maximum_episode_steps <= 0:
            raise ValueError("maximum_episode_steps must be positive")


@dataclass(frozen=True)
class DomainParameters:
    """Per-episode physical parameters with nominal defaults.

    Sampled on reset by SesameEnv._randomize_domain and applied to the MuJoCo
    model; also placed unchanged in info["domain"].

    Attributes:
        mass_scale: Multiplier on the mass and inertia of every robot body.
        friction: Multiplier on the sliding friction coefficient of all geoms.
        servo_strength: Multiplier on the servo maximum force range.
        servo_kp_scale: Multiplier on the position servo stiffness (kp).
        servo_time_constant_s: First-order lag time constant of the servo
            target, in seconds.
        action_delay_steps: Action delay in control steps (0 means no delay).
        imu_noise_scale: IMU noise multiplier; 0 means no noise.
        battery_voltage: Battery voltage in volts; the force range is scaled
            relative to a 7.4 V baseline.
        com_offset_m: Base center-of-mass offset in meters, added to the
            base's body_ipos.
    """

    mass_scale: float = 1.0
    friction: float = 1.0
    servo_strength: float = 1.0
    servo_kp_scale: float = 1.0
    servo_time_constant_s: float = 0.05
    action_delay_steps: int = 0
    imu_noise_scale: float = 0.0
    battery_voltage: float = 7.4
    com_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


class SesameEnv(gym.Env):
    """Gymnasium environment driving Sesame with 8 residual actions at 50 Hz.

    The observation is the 43-dimensional proprioception (slices in the module
    docstring), optionally with the front camera image, which is held at
    camera_hz. The environment owns the MuJoCo model and data plus lazily
    created renderers/viewer; call close() to release them when done.

    Attributes:
        task: Current task.
        config: Configuration with constructor overrides applied.
        frame_skip: Number of physics steps per control step.
        model: MuJoCo model; domain randomization is applied in place on reset.
        data: MuJoCo simulation data.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task: Task | str | None = None,
        config: SesameEnvConfig | None = None,
        render_mode: str | None = None,
        domain_randomization: bool | None = None,
    ) -> None:
        """Creates the environment and resolves the model object ids it needs.

        Args:
            task: Task; None uses config.task.
            config: Environment configuration; None uses the default.
            render_mode: None, "human" or "rgb_array".
            domain_randomization: If not None, overrides the field of the same
                name in config.

        Raises:
            ValueError: render_mode is unsupported, or the model lacks a
                required joint, body, site, geom or sensor.
        """

        super().__init__()
        config = config or SesameEnvConfig()
        self.task = Task(task) if task is not None else Task(config.task)
        updates: dict[str, Any] = {"task": self.task}
        if domain_randomization is not None:
            updates["domain_randomization"] = domain_randomization
        self.config = replace(config, **updates)
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": round(1 / self.config.control_dt)}
        self.model = load_model()
        self.model.opt.timestep = self.config.physics_dt
        # The offscreen buffer must fit the larger of the camera and external
        # rendering resolutions.
        self.model.vis.global_.offwidth = max(self.config.render_width, self.config.camera_width)
        self.model.vis.global_.offheight = max(self.config.render_height, self.config.camera_height)
        self.data = make_data(self.model)
        self.frame_skip = round(self.config.control_dt / self.config.physics_dt)
        self._joint_qpos_adr = joint_qpos_addresses(self.model)
        self._joint_dof_adr = joint_dof_addresses(self.model)
        # Floating-base qpos/dof addresses and IMU ids, used for state reads
        # and reset.
        root = self._id(mujoco.mjtObj.mjOBJ_JOINT, "root")
        self._root_qpos_adr = int(self.model.jnt_qposadr[root])
        self._root_dof_adr = int(self.model.jnt_dofadr[root])
        self._base_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "base")
        self._imu_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "imu")
        self._target_mocap_id = int(
            self.model.body_mocapid[self._id(mujoco.mjtObj.mjOBJ_BODY, "target")]
        )
        # Navigation-only objects (target marker and obstacles); hidden during
        # domain randomization for other tasks.
        self._obstacle_body_ids = np.asarray(
            [self._id(mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}") for i in range(2)]
        )
        self._navigation_geom_ids = np.asarray(
            [self._id(mujoco.mjtObj.mjOBJ_GEOM, "target_visual")]
            + np.flatnonzero(np.isin(self.model.geom_bodyid, self._obstacle_body_ids)).tolist()
        )
        # Bodies with mass > 0 count as the robot; used to scale mass and to
        # detect obstacle contact.
        self._robot_body_ids = np.flatnonzero(self.model.body_mass > 0)
        self._sensor_slices: dict[str, slice] = {}
        for name in (
            "imu_quat",
            "imu_gyro",
            "imu_accel",
            "imu_velocity",
            "r3_touch",
            "r4_touch",
            "l3_touch",
            "l4_touch",
        ):
            sensor = self._id(mujoco.mjtObj.mjOBJ_SENSOR, name)
            address = int(self.model.sensor_adr[sensor])
            self._sensor_slices[name] = slice(address, address + int(self.model.sensor_dim[sensor]))
        # Keep the nominal model parameters so every reset reapplies domain
        # randomization from a clean baseline; scales and offsets must not
        # accumulate across episodes.
        self._nominal = {
            name: getattr(self.model, name).copy()
            for name in (
                "body_mass",
                "body_inertia",
                "body_ipos",
                "body_pos",
                "geom_friction",
                "geom_rgba",
                "geom_contype",
                "geom_conaffinity",
                "actuator_gainprm",
                "actuator_biasprm",
                "actuator_forcerange",
            )
        }
        # The action is the normalized residual of the 8 joints (firmware
        # order).
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        proprio = spaces.Box(-np.inf, np.inf, shape=(43,), dtype=np.float32)
        self.observation_space = (
            spaces.Dict(
                {
                    "proprio": proprio,
                    "rgb": spaces.Box(
                        0,
                        255,
                        shape=(self.config.camera_height, self.config.camera_width, 3),
                        dtype=np.uint8,
                    ),
                }
            )
            if self.config.observation_mode == "pixels"
            else proprio
        )
        # Render resources are created lazily and released together by close().
        self._camera_renderer = None
        self._external_renderer = None
        self._viewer = None
        self._camera_frame: np.ndarray | None = None
        self._next_camera_time = 0.0
        # Episode state; the remaining counters are initialized in reset() and
        # must not be used before the first reset.
        self._domain = DomainParameters()
        self._servo_targets = STAND_ANGLES_RAD.copy()
        self._last_action = np.zeros(8)
        self._command = np.zeros(3)
        self._goal = np.zeros(3)
        self._target_history: deque[np.ndarray] = deque(maxlen=4)
        self._has_reset = False
        self._episode_done = False

    def _id(self, kind: mujoco.mjtObj, name: str) -> int:
        """Returns a MuJoCo object id by name; raises ValueError if missing."""

        result = mujoco.mj_name2id(self.model, kind, name)
        if result < 0:
            raise ValueError(f"Model is missing {kind.name} {name!r}")
        return result

    @property
    def servo_targets(self) -> np.ndarray:
        """Current servo targets (radians, firmware order); returns a copy.

        These are the targets actually sent to the actuators after delay,
        first-order lag and rate limiting, not the target the policy requested
        this step.
        """

        return self._servo_targets.copy()

    @property
    def true_joint_positions(self) -> np.ndarray:
        """True joint positions (radians, firmware order); returns a copy.

        Read from simulation state; not directly available on the deployed
        robot.
        """

        return self.data.qpos[self._joint_qpos_adr].copy()

    def training_view_state(self) -> np.ndarray:
        """Returns a copy of the current raw MuJoCo qpos.

        Intended for the parent process's live display. The shape is
        (model.nq,) in raw MuJoCo qpos layout, not the 8-joint firmware order:
        the first 7 elements are the floating base joint, with position
        (x, y, z) in meters and the orientation quaternion in wxyz order;
        the rest are the hinge joint angles in radians. This method only
        reads state: it does not change physics, servo targets or random
        number state, and modifying the returned array does not affect the
        environment.

        Returns:
            An independent copy of the current data.qpos.

        Raises:
            gym.error.ResetNeeded: reset() has not been called.
        """

        if not self._has_reset:
            raise gym.error.ResetNeeded("Call reset() before reading the training view state")
        return self.data.qpos.copy()

    @property
    def command(self) -> np.ndarray:
        """Current velocity command [forward m/s, lateral m/s, yaw rad/s].

        Returns a copy.
        """

        return self._command.copy()

    @property
    def goal(self) -> np.ndarray:
        """Navigation goal position (world frame, meters); returns a copy.

        Zeros for non-navigation tasks.
        """

        return self._goal.copy()

    @property
    def maximum_episode_steps(self) -> int:
        """Configured episode step limit, or the task default."""

        return self.config.maximum_episode_steps or self.task.default_episode_steps

    def _sensor(self, name: str) -> np.ndarray:
        """View of a named sensor's data.

        Modifying the returned array also modifies sensordata.
        """

        return self.data.sensordata[self._sensor_slices[name]]

    @property
    def language_instruction(self) -> str:
        """Instruction text for language-conditioned policies for this task."""

        return {
            Task.STAND: "Stand still.",
            Task.RECOVERY: "Recover to a stable standing pose.",
            Task.LOCOMOTION: "Follow the velocity and turning command.",
            Task.NAVIGATION: "Walk to the green marker without touching the obstacles.",
        }[self.task]

    def _update_imu(self) -> None:
        """Samples the IMU reading shared by this control step.

        Sampled once per control step (and on reset); the observation and
        deployment_imu() share the same reading. Noise is scaled by
        imu_noise_scale: each component of the attitude perturbation vector
        has a normal standard deviation of 0.01 rad, the gyroscope 0.015 rad/s
        and the acceleration 0.03 g. Reward and termination do not use this
        noisy sample.
        """

        quaternion = self._sensor("imu_quat").copy()
        gyro = self._sensor("imu_gyro").copy()
        acceleration = self._sensor("imu_accel").copy()
        noise = self._domain.imu_noise_scale
        if noise:
            axis = self.np_random.normal(0, 0.01 * noise, 3)
            angle = float(np.linalg.norm(axis))
            if angle > 0:
                perturbation = np.empty(4)
                mujoco.mju_axisAngle2Quat(perturbation, axis / angle, angle)
                noisy = np.empty(4)
                mujoco.mju_mulQuat(noisy, quaternion, perturbation)
                quaternion = noisy
            gyro += self.np_random.normal(0, 0.015 * noise, 3)
            acceleration += self.np_random.normal(0, 0.03 * 9.81 * noise, 3)
        self._imu = (quaternion, gyro, acceleration)

    def deployment_imu(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns copies of the noisy IMU reading shared this step.

        Returns:
            A tuple of three array copies: (quaternion wxyz, gyroscope in
            rad/s, acceleration in m/s^2), with shapes (4,), (3,) and (3,).
            The contents come from the same sample the observation uses.

        Raises:
            gym.error.ResetNeeded: reset() has not been called.
        """

        if not self._has_reset:
            raise gym.error.ResetNeeded("Call reset() before reading the IMU")
        return tuple(value.copy() for value in self._imu)

    def _rotation(self) -> np.ndarray:
        """Body-to-world rotation matrix of the IMU site (3x3, noise-free)."""

        return self.data.site_xmat[self._imu_site_id].reshape(3, 3)

    def _up(self) -> np.ndarray:
        """Direction of world +z in the body frame; z is the uprightness."""

        return self._rotation()[2, :]

    def _contacts(self) -> np.ndarray:
        """Touch flags of the four feet (r3, r4, l3, l4) as 0/1 float32."""

        return np.asarray(
            [self._sensor(f"{foot}_touch")[0] > 1e-5 for foot in ("r3", "r4", "l3", "l4")],
            dtype=np.float32,
        )

    def _randomize_domain(self) -> None:
        """Restores the model and applies this episode's domain parameters.

        First restores every modified array from self._nominal, then samples
        the parameters (or uses the nominal ones) according to the
        configuration, and finally calls mj_setConst so derived constants
        match the new parameters.
        """

        for name, nominal in self._nominal.items():
            getattr(self.model, name)[:] = nominal
        rng = self.np_random
        if self.config.domain_randomization:
            self._domain = DomainParameters(
                mass_scale=float(rng.uniform(0.85, 1.15)),
                friction=float(rng.uniform(0.65, 1.3)),
                servo_strength=float(rng.uniform(0.8, 1.15)),
                servo_kp_scale=float(rng.uniform(0.8, 1.2)),
                servo_time_constant_s=float(rng.uniform(0.025, 0.08)),
                action_delay_steps=int(rng.integers(0, 4)),
                imu_noise_scale=float(rng.uniform(0.5, 1.5)),
                battery_voltage=float(rng.uniform(6.4, 8.4)),
                com_offset_m=tuple(float(x) for x in rng.uniform(-0.003, 0.003, 3)),
            )
        else:
            self._domain = DomainParameters()
        domain = self._domain
        self.model.body_mass[self._robot_body_ids] *= domain.mass_scale
        self.model.body_inertia[self._robot_body_ids] *= domain.mass_scale
        self.model.body_ipos[self._base_body_id] += domain.com_offset_m
        self.model.geom_friction[:, 0] *= domain.friction
        # The position servo's gainprm[0] (kp) and biasprm[1] (-kp) must be
        # scaled as a pair; otherwise the actuator is no longer consistent
        # position control.
        self.model.actuator_gainprm[:, 0] *= domain.servo_kp_scale
        self.model.actuator_biasprm[:, 1] *= domain.servo_kp_scale
        # Battery voltage is relative to a 7.4 V baseline and scales the force
        # range together with the servo strength.
        self.model.actuator_forcerange[:] *= domain.servo_strength * domain.battery_voltage / 7.4
        # For non-navigation tasks, hide the target and obstacles and disable
        # their collisions.
        if self.task is not Task.NAVIGATION:
            self.model.geom_rgba[self._navigation_geom_ids, 3] = 0
            self.model.geom_contype[self._navigation_geom_ids] = 0
            self.model.geom_conaffinity[self._navigation_geom_ids] = 0
        mujoco.mj_setConst(self.model, self.data)

    def _sample_command(self) -> None:
        """Samples a locomotion command and schedules its next change."""

        # Every locomotion episode starts with a real tracking challenge;
        # otherwise a near-zero command could let a policy that stands still
        # be misjudged as passing.
        forward = self.np_random.random() < 2 / 3
        speed = self.np_random.uniform(0.08, 0.24 if forward else 0.12)
        self._command[:] = [
            speed if forward else -speed,
            self.np_random.uniform(-0.06, 0.06),
            self.np_random.uniform(-1.2, 1.2),
        ]
        # Mid-episode, with 15% probability switch to an all-zero command
        # (asking the robot to stop); this never happens at episode start.
        if self._steps > 0 and self.np_random.random() < 0.15:
            self._command[:] = 0
        self._next_command_step = self._steps + max(
            1, round(self.np_random.uniform(2.5, 6.0) / self.config.control_dt)
        )

    def _navigation_command(self) -> None:
        """Computes the navigation velocity command from the goal bearing.

        The bearing is taken in the body frame. Forward speed is proportional
        to distance (capped at 0.18 m/s), shrinks with the heading error, and
        is 0 when facing away from the goal; no lateral motion is used; yaw
        rate is 2 times the bearing, limited to +-1.2 rad/s.
        """

        relative = self._rotation().T @ (
            self._goal - self.data.qpos[self._root_qpos_adr : self._root_qpos_adr + 3]
        )
        bearing = math.atan2(relative[1], relative[0])
        self._command[:] = [
            min(0.18, float(np.linalg.norm(relative[:2]))) * max(0.0, math.cos(bearing)),
            0.0,
            np.clip(2 * bearing, -1.2, 1.2),
        ]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Resets the simulation and initializes a new task episode.

        Resets the state, then sets the initial conditions for the task.

        Args:
            seed: Random seed; None keeps the current random number generator.
            options: Unused; present only to match the Gymnasium interface.

        Returns:
            (observation, info). For locomotion the initial command is already
            sampled, and for navigation the command is computed from the
            initial pose; it appears in both the observation and
            info["command"].
        """

        super().reset(seed=seed)
        self._randomize_domain()
        set_stand_pose(self.model, self.data)
        self._steps = 0
        self._return = 0.0
        self._energy = 0.0
        self._distance = 0.0
        self._obstacle_contact_steps = 0
        self._stable_steps = 0
        self._upright_sum = 0.0
        self._linear_error_sum = 0.0
        self._yaw_error_sum = 0.0
        self._command[:] = 0
        self._goal[:] = 0
        self._last_action[:] = 0
        self._servo_targets[:] = STAND_ANGLES_RAD
        self._target_history.clear()
        self._target_history.extend(STAND_ANGLES_RAD.copy() for _ in range(4))
        self._camera_frame = None
        self._next_camera_time = 0.0
        self._episode_done = False
        self._has_reset = True
        if self.task is Task.NAVIGATION:
            self._goal[:] = [
                self.np_random.uniform(0.8, 1.4),
                self.np_random.uniform(-0.6, 0.6),
                0.035,
            ]
            self.data.mocap_pos[self._target_mocap_id] = self._goal
            if self.config.domain_randomization:
                self.model.body_pos[self._obstacle_body_ids, :2] += self.np_random.uniform(
                    -0.12, 0.12, (2, 2)
                )
                mujoco.mj_setConst(self.model, self.data)
                set_stand_pose(self.model, self.data)
                self.data.mocap_pos[self._target_mocap_id] = self._goal
        # Recovery task: roll about the x axis by 0.9-1.7 rad (random side),
        # starting directly from the fallen pose without stand settling.
        if self.task is Task.RECOVERY:
            roll = float(self.np_random.uniform(0.9, 1.7) * self.np_random.choice([-1, 1]))
            self.data.qpos[self._root_qpos_adr + 3 : self._root_qpos_adr + 7] = [
                math.cos(roll / 2),
                math.sin(roll / 2),
                0,
                0,
            ]
            self.data.qpos[self._root_qpos_adr + 2] = 0.14
            mujoco.mj_forward(self.model, self.data)
            # Lift the whole tilted collision shape above the ground to avoid
            # initial penetration.
            for _ in range(20):
                penetration = max(
                    (-float(c.dist) for c in self.data.contact[: self.data.ncon]), default=0
                )
                if penetration <= 0:
                    break
                self.data.qpos[self._root_qpos_adr + 2] += penetration + 0.001
                mujoco.mj_forward(self.model, self.data)
        else:
            mujoco.mj_forward(self.model, self.data)
            for _ in range(round(0.25 / self.config.physics_dt)):
                mujoco.mj_step(self.model, self.data)
        # Episode time excludes the reset settling; recompute the physical
        # sensors at the final state.
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._previous_xy = self.data.qpos[self._root_qpos_adr : self._root_qpos_adr + 2].copy()
        if self.task is Task.LOCOMOTION:
            self._sample_command()
        elif self.task is Task.NAVIGATION:
            self._navigation_command()
        self._previous_goal_distance = self._goal_distance()
        self._update_imu()
        observation = self._observation()
        info = self._info(success=False, fall=False, out_of_bounds=False)
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        """Advances one control step with a residual action.

        Args:
            action: Normalized residual action of length 8 in firmware joint
                order; values outside [-1, 1] are clipped. The servo target is
                STAND_ANGLES_RAD + action_scale_rad * action, then clipped to
                the servo mechanical limits.

        Returns:
            (observation, reward, terminated, truncated, info);
            info["command"] is the command used for this step, while the
            observation holds the next step's command.

        Raises:
            ValueError: action does not have shape (8,) or contains
                non-finite values.
            gym.error.ResetNeeded: reset() has not been called, or the
                previous episode has ended.
        """

        action = np.clip(validate_joint_vector(action, name="action"), -1, 1)
        target = clip_servo_targets(STAND_ANGLES_RAD + self.config.action_scale_rad * action)
        return self._advance(target, action)

    def step_servo_targets(self, target_rad):
        """Advances one control step with absolute servo targets (radians).

        Intended for deployment replay and comparison.

        Args:
            target_rad: Absolute targets of length 8 in firmware joint order
                (radians); clipped to the servo mechanical limits.

        Returns:
            Same as step(). The reported action (used for the observation,
            the previous action and the action-rate reward) is back-computed
            from the target and clipped to [-1, 1]; the actual target is not
            affected by this clipping.

        Raises:
            ValueError: target_rad does not have shape (8,) or contains
                non-finite values.
            gym.error.ResetNeeded: reset() has not been called, or the
                previous episode has ended.
        """

        target = clip_servo_targets(target_rad)
        action = np.clip((target - STAND_ANGLES_RAD) / self.config.action_scale_rad, -1, 1)
        return self._advance(target, action)

    def _advance(self, desired: np.ndarray, action: np.ndarray):
        """Shared flow: servo dynamics, physics, termination, success, reward.

        Args:
            desired: Absolute servo target requested by the policy (radians,
                already clipped).
            action: Corresponding normalized action, used only for the
                observation and the action-rate penalty.

        Returns:
            (observation, reward, terminated, truncated, info).
            terminated means a fall, out of bounds, reaching the goal or
            recovery completed; truncated holds only when the step limit is
            reached without termination.

        Raises:
            gym.error.ResetNeeded: reset() has not been called, or the
                previous episode has ended.
        """

        if not self._has_reset or self._episode_done:
            raise gym.error.ResetNeeded("Call reset() before stepping a new episode")
        # Servo model: first apply the action delay (a domain parameter), then
        # approach the target with a first-order lag, limited by the servo
        # maximum speed, and finally clip to the joint limits.
        self._target_history.append(desired.copy())
        delayed = self._target_history[-1 - self._domain.action_delay_steps]
        alpha = -math.expm1(-self.config.control_dt / self._domain.servo_time_constant_s)
        maximum_delta = SERVO_MAX_SPEED_RAD_S * self.config.control_dt
        self._servo_targets += np.clip(
            alpha * (delayed - self._servo_targets), -maximum_delta, maximum_delta
        )
        np.clip(
            self._servo_targets,
            JOINT_LIMITS_RAD[:, 0],
            JOINT_LIMITS_RAD[:, 1],
            out=self._servo_targets,
        )
        self.data.ctrl[:] = self._servo_targets
        # Accumulate mechanical work (|torque * angular velocity|) and obstacle
        # contact at every physics step.
        energy = 0.0
        collision = False
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            energy += (
                float(
                    np.sum(np.abs(self.data.actuator_force * self.data.qvel[self._joint_dof_adr]))
                )
                * self.config.physics_dt
            )
            collision = collision or self._has_obstacle_contact()
        # The statistics below all come from the noise-free simulation state,
        # not from the noisy IMU sample.
        self._steps += 1
        self._energy += energy
        self._obstacle_contact_steps += int(collision)
        xy = self.data.qpos[self._root_qpos_adr : self._root_qpos_adr + 2]
        self._distance += float(np.linalg.norm(xy - self._previous_xy))
        self._previous_xy[:] = xy
        upright = float(self._up()[2])
        height = float(self.data.qpos[self._root_qpos_adr + 2])
        velocity = self._sensor("imu_velocity")
        gyro = self._sensor("imu_gyro")
        linear_error = float(np.sum((velocity[:2] - self._command[:2]) ** 2))
        yaw_error = float((gyro[2] - self._command[2]) ** 2)
        self._upright_sum += upright
        self._linear_error_sum += linear_error
        self._yaw_error_sum += yaw_error
        # Fall: non-finite state, or (except for recovery) too low or tilted
        # too far; the recovery task starts from a fallen pose, so it does not
        # use pose to detect falls.
        finite = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        fall = not finite or (self.task is not Task.RECOVERY and (height < 0.023 or upright < 0.15))
        out_of_bounds = self.task is Task.NAVIGATION and bool(np.any(np.abs(xy) > 3.0))
        # Stable standing: high enough, nearly upright and low angular
        # velocity; the consecutive-stable count determines recovery success.
        stable = height > 0.035 and upright > 0.9 and np.linalg.norm(gyro) < 0.5
        self._stable_steps = self._stable_steps + 1 if stable else 0
        reached = self.task is Task.NAVIGATION and self._goal_distance() <= 0.12
        recovered = (
            self.task is Task.RECOVERY and self._stable_steps * self.config.control_dt >= 0.5
        )
        # Termination and truncation are mutually exclusive; a step that
        # reaches the limit but already terminated does not count as truncated.
        terminated = bool(fall or out_of_bounds or reached or recovered)
        truncated = bool(self._steps >= self.maximum_episode_steps and not terminated)
        # Strict success: the stand task only needs to last to the limit;
        # locomotion at the limit requires mean uprightness >= 0.75, linear
        # velocity RMS error <= 0.06 m/s and yaw RMS error <= 0.45 rad/s;
        # navigation must reach the goal and recovery must be stable for
        # 0.5 s, and none may fall or go out of bounds.
        success = bool(not fall and not out_of_bounds and (reached or recovered))
        if truncated and self.task is Task.STAND:
            success = True
        elif truncated and self.task is Task.LOCOMOTION:
            success = bool(
                self._upright_sum / self._steps >= 0.75
                and math.sqrt(self._linear_error_sum / self._steps) <= 0.06
                and math.sqrt(self._yaw_error_sum / self._steps) <= 0.45
            )
        # Reward: uprightness + closeness to the stand pose - energy - action
        # rate, plus tracking, recovery height or navigation progress terms by
        # task; after multiplying by control_dt, subtract 2 for a fall or out
        # of bounds and add 2 for recovery completed or goal reached.
        pose_error = float(np.mean((self.true_joint_positions - STAND_ANGLES_RAD) ** 2))
        action_rate = float(np.mean((action - self._last_action) ** 2))
        reward = (
            0.5 * max(0, upright)
            + 0.25 * math.exp(-pose_error / 0.3**2)
            - 0.003 * energy / self.config.control_dt
            - 0.025 * action_rate
        )
        if self.task in (Task.LOCOMOTION, Task.NAVIGATION):
            reward += 1.5 * math.exp(-linear_error / 0.1**2) + 0.55 * math.exp(-yaw_error / 0.45**2)
        elif self.task is Task.RECOVERY:
            reward += 0.5 * math.exp(-(((height - 0.047) / 0.02) ** 2))
        if self.task is Task.NAVIGATION:
            distance = self._goal_distance()
            reward += 5 * (self._previous_goal_distance - distance) / self.config.control_dt
            reward -= 0.4 * collision
            self._previous_goal_distance = distance
        reward = reward * self.config.control_dt - 2 * float(fall or out_of_bounds)
        if recovered or reached:
            reward += 2
        self._return += reward
        self._last_action[:] = action
        # info reports the command used for this transition; the next command
        # (if any) is sampled afterwards and shown with the next observation,
        # so the policy knows the target it is about to track.
        info = self._info(success=success, fall=fall, out_of_bounds=out_of_bounds)
        if self.task is Task.LOCOMOTION and self._steps >= self._next_command_step:
            self._sample_command()
        elif self.task is Task.NAVIGATION:
            self._navigation_command()
        self._episode_done = terminated or truncated
        if self._episode_done:
            info["episode"] = {
                "r": self._return,
                "l": self._steps,
                "fall": fall,
                "success": success,
                "distance_m": self._distance,
                "energy_j": self._energy,
            }
        self._update_imu()
        observation = self._observation()
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

    def _has_obstacle_contact(self) -> bool:
        """Whether a robot-obstacle contact exists in the contact list.

        Only checked for the navigation task.
        """

        if self.task is not Task.NAVIGATION:
            return False
        for contact in self.data.contact[: self.data.ncon]:
            a = int(self.model.geom_bodyid[contact.geom1])
            b = int(self.model.geom_bodyid[contact.geom2])
            if (a in self._obstacle_body_ids and b in self._robot_body_ids) or (
                b in self._obstacle_body_ids and a in self._robot_body_ids
            ):
                return True
        return False

    def _goal_distance(self) -> float:
        """Horizontal distance between the body and the goal (meters)."""

        return float(
            np.linalg.norm(
                self._goal[:2] - self.data.qpos[self._root_qpos_adr : self._root_qpos_adr + 2]
            )
        )

    def _info(self, *, success: bool, fall: bool, out_of_bounds: bool) -> dict[str, Any]:
        """Builds diagnostic info using the command current at call time.

        Args:
            success: Strict success result for this step.
            fall: Whether the robot fell.
            out_of_bounds: Whether the robot left the navigation bounds.

        Returns:
            A dict with the task, physics state, command, domain parameters
            and success/failure flags.
        """

        return {
            "task": self.task.value,
            "instruction": self.language_instruction,
            "upright": float(self._up()[2]),
            "base_height_m": float(self.data.qpos[self._root_qpos_adr + 2]),
            "base_velocity_m_s": self._sensor("imu_velocity").copy(),
            "yaw_rate_rad_s": float(self._sensor("imu_gyro")[2]),
            "command": self.command,
            "servo_target_rad": self.servo_targets,
            "goal_distance_m": self._goal_distance() if self.task is Task.NAVIGATION else 0.0,
            "obstacle_contact_steps": self._obstacle_contact_steps,
            "domain": asdict(self._domain),
            "success": bool(success),
            "fallen": bool(fall),
            "is_success": bool(success),
            "obstacle_contact": self._has_obstacle_contact(),
            "out_of_bounds": bool(out_of_bounds),
        }

    def _observation(self):
        """Builds the 43-value observation described in the module docstring.

        Uses the noisy IMU sample shared this step; outside privileged mode the
        privileged fields stay 0. In pixels mode the front camera image is
        also attached.

        Returns:
            In state mode, a float32 array of shape (43,); in pixels mode, a
            dict with the proprio array and the rgb image.
        """

        state = np.zeros(43, dtype=np.float32)
        quaternion, gyro, acceleration = self.deployment_imu()
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        state[0:3] = rotation.reshape(3, 3)[2, :]
        state[3:6] = acceleration / 9.81
        state[6:9] = gyro
        state[9:17] = self._servo_targets - STAND_ANGLES_RAD
        if self.config.privileged_joint_state:
            state[9:17] = self.true_joint_positions - STAND_ANGLES_RAD
            state[17:25] = self.data.qvel[self._joint_dof_adr] / 10
            state[36:40] = self._contacts()
            if self.task is Task.NAVIGATION:
                relative = self._rotation().T @ (
                    self._goal - self.data.qpos[self._root_qpos_adr : self._root_qpos_adr + 3]
                )
                state[40:43] = relative
        state[25:33] = self._last_action
        state[33:36] = self._command / np.asarray([0.25, 0.25, 1.4])
        if self.config.observation_mode == "state":
            return state
        return {"proprio": state, "rgb": self.render_camera()}

    def render_camera(self) -> np.ndarray:
        """Returns the front camera RGB image (height x width x 3, uint8).

        The image updates at camera_hz in simulation time; between updates a
        copy of the same frame is returned, to mimic the low update rate of a
        real camera. The renderer is created on first use.

        Raises:
            gym.error.ResetNeeded: reset() has not been called.
        """

        if not self._has_reset:
            raise gym.error.ResetNeeded("Call reset() before rendering the camera")
        if self._camera_frame is None or self.data.time + 1e-9 >= self._next_camera_time:
            if self._camera_renderer is None:
                self._camera_renderer = mujoco.Renderer(
                    self.model, height=self.config.camera_height, width=self.config.camera_width
                )
            self._camera_renderer.update_scene(self.data, camera="front_camera")
            self._camera_frame = self._camera_renderer.render().copy()
            self._next_camera_time = (
                math.floor((self.data.time + 1e-9) * self.config.camera_hz) + 1
            ) / self.config.camera_hz
        return self._camera_frame.copy()

    def render(self):
        """Renders a frame according to render_mode.

        Returns:
            In "rgb_array" mode, the RGB image from the external viewpoint; in
            "human" mode, syncs the passive viewer and returns None; None when
            render_mode is not set.
        """

        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer

                self._viewer = viewer.launch_passive(self.model, self.data)
            if self._viewer.is_running():
                self._viewer.sync()
            return None
        if self.render_mode != "rgb_array":
            return None
        if self._external_renderer is None:
            self._external_renderer = mujoco.Renderer(
                self.model, height=self.config.render_height, width=self.config.render_width
            )
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        base = self.data.qpos[self._root_qpos_adr : self._root_qpos_adr + 3]
        camera.lookat[:] = base
        camera.distance = 0.55
        camera.azimuth = 135
        camera.elevation = -25
        if self.task is Task.NAVIGATION:
            camera.lookat[:] = (base + self._goal) / 2
            camera.distance = max(0.7, self._goal_distance() * 1.5)
            camera.elevation = -45
        self._external_renderer.update_scene(self.data, camera=camera)
        return self._external_renderer.render().copy()

    def close(self) -> None:
        """Releases the camera/external renderers and viewer; safe to repeat."""

        for name in ("_camera_renderer", "_external_renderer", "_viewer"):
            resource = getattr(self, name)
            if resource is not None:
                resource.close()
                setattr(self, name, None)
