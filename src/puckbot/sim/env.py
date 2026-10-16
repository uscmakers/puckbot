"""The air hockey world: reset it, give the paddle a target, step time forward.

Steps (every control tick):
    1. Command   PaddleMotion moves the commanded point toward the target (speed + accel limits)
    2. Physics   MuJoCo runs several 1 ms ticks; a spring-damper pulls the paddle after that point
    3. Report    return a State: where the puck and paddle are, and whether a goal was scored

Usage:
    env = AirHockeyEnv()
    state = env.reset(puck_pos=(-0.4, 0.1))
    state = env.step(target=(-0.5, 0.1))      # one control tick (20 ms at 50 Hz)
    state.puck_pos, state.paddle_pos, state.goal

The robot defends the -x goal and attacks the +x goal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from puckbot.sim.model import TableGeometry, build_model, paddle_limits
from puckbot.sim.paddle_motion import PaddleMotion
from puckbot.sim.settings import Settings, load_settings

Goal = Literal["scored", "conceded"]  # scored = puck left through the +x slot; conceded = the -x slot


# ── Output ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class State:
    time: float
    puck_pos: np.ndarray  # (x, y) metres
    puck_vel: np.ndarray
    paddle_pos: np.ndarray
    paddle_vel: np.ndarray
    goal: Goal | None


# ── World ─────────────────────────────────────────────────────────────────────


class AirHockeyEnv:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.model = build_model(self.settings)
        self.data = mujoco.MjData(self.model)
        self.geometry = TableGeometry.from_model(self.model)

        low, high = paddle_limits(self.settings, self.geometry)
        self.paddle_low, self.paddle_high = np.array(low), np.array(high)
        self.home = np.array(self.settings.paddle.home_position_m)
        self._motion = PaddleMotion(self.settings.paddle, low, high, start=self.home)

        self.physics_dt = float(self.model.opt.timestep)
        self.physics_ticks_per_step = round(1 / (self.settings.control_rate_hz * self.physics_dt))
        self.step_seconds = self.physics_ticks_per_step * self.physics_dt

        self._puck = _joint_slots(self.model, ["puck_x", "puck_y"])
        self._paddle = _joint_slots(self.model, ["paddle_x", "paddle_y"])

    # ── Reset and step ──

    def reset(self, puck_pos=(0.0, 0.0), puck_vel=(0.0, 0.0), paddle_pos=None) -> State:
        paddle_pos = self.home if paddle_pos is None else np.asarray(paddle_pos, dtype=float)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._puck.position] = puck_pos
        self.data.qvel[self._puck.velocity] = puck_vel
        self.data.qpos[self._paddle.position] = paddle_pos
        self._motion.reset(paddle_pos)
        mujoco.mj_forward(self.model, self.data)
        return self.state

    def step(self, target) -> State:
        """Run one control tick with the paddle heading for `target` = (x, y) metres."""
        for _ in range(self.physics_ticks_per_step):
            self._motion.advance(target, self.physics_dt)
            self.data.qfrc_applied[self._paddle.velocity] = self._pull_on_paddle()
            mujoco.mj_step(self.model, self.data)
        return self.state

    def _pull_on_paddle(self) -> np.ndarray:
        """Force (x, y) that drags the paddle along the commanded point, capped at max_force_n.

        Spring on the position error + damper on the velocity error, so it follows a moving point
        with almost no lag. A hard hit by the puck can still push the paddle off its path.
        """
        paddle = self.settings.paddle
        position_error = self._motion.position - self.data.qpos[self._paddle.position]
        velocity_error = self._motion.velocity - self.data.qvel[self._paddle.velocity]
        force = paddle.stiffness_n_per_m * position_error + paddle.damping_ns_per_m * velocity_error
        return np.clip(force, -paddle.max_force_n, paddle.max_force_n)

    # ── Read ──

    @property
    def paddle_target(self) -> np.ndarray:
        """The point the paddle is being pulled toward right now (after speed + accel limits)."""
        return self._motion.position.copy()

    @property
    def state(self) -> State:
        puck_pos = self.data.qpos[self._puck.position].copy()
        return State(
            time=float(self.data.time),
            puck_pos=puck_pos,
            puck_vel=self.data.qvel[self._puck.velocity].copy(),
            paddle_pos=self.data.qpos[self._paddle.position].copy(),
            paddle_vel=self.data.qvel[self._paddle.velocity].copy(),
            goal=self._goal_at(puck_pos),
        )

    def _goal_at(self, puck_pos) -> Goal | None:
        x, y = puck_pos
        in_slot = abs(y) < self.geometry.goal_half_width
        if in_slot and x > self.geometry.goal_exit_x:
            return "scored"
        if in_slot and x < -self.geometry.goal_exit_x:
            return "conceded"
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _JointSlots:
    """Where a pair of joints live in MuJoCo's qpos / qvel arrays."""

    position: list[int]
    velocity: list[int]


def _joint_slots(model: mujoco.MjModel, names: list[str]) -> _JointSlots:
    joints = [model.joint(name) for name in names]
    return _JointSlots(
        position=[int(j.qposadr[0]) for j in joints],
        velocity=[int(j.dofadr[0]) for j in joints],
    )
