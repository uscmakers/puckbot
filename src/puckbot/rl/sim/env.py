"""The air hockey world: reset it, give each paddle a target, step time forward.

Steps (every control tick):
    1. Command   each paddle gets an (x, y) target; its commanded point moves toward it within the
                 speed and acceleration limits, braking so it stops on the target (PaddleMotion)
    2. Physics   MuJoCo runs several physics ticks: each paddle is a mass pulled after the commanded
                 point by a critically damped spring (MuJoCo position actuator); pucks slide and bounce
    3. Report    return a State: where everything is, whether a goal was scored, who touched what

The Layout says how many paddles and pucks there are and where each paddle may go. Nothing else in
the physics depends on it. Paddle 0 is the robot's; it defends the -x goal and attacks the +x goal.

Usage:
    env = AirHockeyEnv()                          # one robot paddle, one puck
    env = AirHockeyEnv(layout=Layout.of(2))       # + an opponent paddle
    state = env.reset(puck_pos=(-0.4, 0.1))
    state = env.step((-0.5, 0.1))                 # one control tick (20 ms): paddle target (x, y) metres
    state = env.step([(-0.5, 0.1), (0.5, 0.0)])   # one (x, y) per paddle
    state.puck_pos        # (pucks, 2) metres       state.paddle_pos    # (paddles, 2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from puckbot.rl.sim.layout import Layout
from puckbot.rl.sim.model import (
    TableGeometry,
    build_model,
    paddle_geom,
    paddle_home,
    paddle_joints,
    paddle_limits,
    puck_geom,
    puck_joints,
)
from puckbot.rl.sim.paddle_motion import PaddleMotion
from puckbot.rl.sim.settings import Settings, load_settings

Goal = Literal["scored", "conceded"]  # scored = puck left through the +x slot; conceded = the -x slot

DEFAULT_PUCK_SPACING_M = 0.10  # extra pucks start in a row across the centre line this far apart


# ── Output ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class State:
    time: float
    puck_pos: np.ndarray  # (pucks, 2) x, y in metres
    puck_vel: np.ndarray
    paddle_pos: np.ndarray  # (paddles, 2)
    paddle_vel: np.ndarray
    goals: tuple[Goal | None, ...]  # one per puck
    touches: np.ndarray  # (paddles, pucks) bool: did this paddle touch this puck during the last tick

    @property
    def goal(self) -> Goal | None:
        """The first goal any puck scored this tick, or None."""
        return next((goal for goal in self.goals if goal), None)


# ── World ─────────────────────────────────────────────────────────────────────


class AirHockeyEnv:
    def __init__(self, settings: Settings | None = None, layout: Layout | None = None):
        self.settings = settings or load_settings()
        self.layout = layout or Layout()
        self.model = build_model(self.settings, self.layout)
        self.data = mujoco.MjData(self.model)
        self.geometry = TableGeometry.from_model(self.model)
        self.n_pucks = self.layout.pucks
        self.n_paddles = len(self.layout.paddles)

        limits = [paddle_limits(setup, self.settings, self.geometry) for setup in self.layout.paddles]
        self.paddle_low = np.array([low for low, _ in limits])  # (paddles, 2)
        self.paddle_high = np.array([high for _, high in limits])
        self.home = np.array(
            [
                paddle_home(setup, self.settings, lims)
                for setup, lims in zip(self.layout.paddles, limits, strict=True)
            ]
        )

        self.physics_dt = float(self.model.opt.timestep)
        self.physics_ticks_per_step = round(1 / (self.settings.control_rate_hz * self.physics_dt))
        self.step_seconds = self.physics_ticks_per_step * self.physics_dt

        self._puck_qpos, self._puck_dof = _joint_slots(
            self.model, [puck_joints(i) for i in range(self.n_pucks)]
        )
        self._paddle_qpos, self._paddle_dof = _joint_slots(
            self.model, [paddle_joints(i) for i in range(self.n_paddles)]
        )
        self._motions = [
            PaddleMotion(self.settings.paddle, low, high, start=home)
            for low, high, home in zip(self.paddle_low, self.paddle_high, self.home, strict=True)
        ]
        self._ctrl_slots = [(2 * i, 2 * i + 1) for i in range(self.n_paddles)]  # the springs, in joint order
        self._ctrl_slots = [
            (2 * i, 2 * i + 1) for i in range(self.n_paddles)
        ]  # the actuators, in joint order
        self._puck_of_geom = _owner_lookup(self.model, [[puck_geom(i)] for i in range(self.n_pucks)])
        self._paddle_of_geom = _owner_lookup(
            self.model, [[paddle_geom(i, "base"), paddle_geom(i, "grip")] for i in range(self.n_paddles)]
        )

        # Positions after every physics tick of the last control tick (a camera reads from these)
        self._qpos_trace = np.zeros((self.physics_ticks_per_step, self.model.nq))
        self._touches = np.zeros((self.n_paddles, self.n_pucks), dtype=bool)

    # ── Reset and step ──

    def reset(self, puck_pos=None, puck_vel=None, paddle_pos=None) -> State:
        """Start over. Each argument is one (x, y) row per puck / paddle (a single row may be flat)."""
        puck_pos = _rows(puck_pos, self.n_pucks, _default_puck_positions(self.n_pucks), "puck_pos")
        puck_vel = _rows(puck_vel, self.n_pucks, np.zeros((self.n_pucks, 2)), "puck_vel")
        paddle_pos = _rows(paddle_pos, self.n_paddles, self.home, "paddle_pos")

        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = paddle_pos.ravel()  # the springs start holding the paddles where they are
        for motion, position in zip(self._motions, paddle_pos, strict=True):
            motion.reset(position)
        self.data.qpos[self._puck_qpos] = puck_pos
        self.data.qvel[self._puck_dof] = puck_vel
        self.data.qpos[self._paddle_qpos] = paddle_pos
        mujoco.mj_forward(self.model, self.data)
        self._qpos_trace[:] = self.data.qpos
        self._touches[:] = False
        return self.state

    def step(self, targets) -> State:
        """Run one control tick. `targets`: one (x, y) metre target per paddle (flat if one paddle)."""
        goals = _rows(targets, self.n_paddles, None, "targets").tolist()
        qpos, ctrl = self.data.qpos, self.data.ctrl
        paddles = list(zip(self._motions, goals, self._ctrl_slots, strict=True))
        self._touches[:] = False

        for tick in range(self.physics_ticks_per_step):
            for motion, (goal_x, goal_y), (x_slot, y_slot) in paddles:
                motion.advance(goal_x, goal_y, self.physics_dt)
                ctrl[x_slot], ctrl[y_slot] = motion.x, motion.y  # each paddle's spring pulls it there
            mujoco.mj_step(self.model, self.data)
            self._qpos_trace[tick] = qpos
            if self.data.ncon:
                self._note_touches()
        return self.state

    def _note_touches(self) -> None:
        count = self.data.ncon
        first, second = self.data.contact.geom1[:count], self.data.contact.geom2[:count]
        for paddle_geom_ids, puck_geom_ids in ((first, second), (second, first)):
            paddles, pucks = self._paddle_of_geom[paddle_geom_ids], self._puck_of_geom[puck_geom_ids]
            both = (paddles >= 0) & (pucks >= 0)
            self._touches[paddles[both], pucks[both]] = True

    # ── Read ──

    @property
    def paddle_target(self) -> np.ndarray:
        """(paddles, 2): the point each paddle's spring is pulling it toward right now."""
        return np.array([motion.position for motion in self._motions])

    @property
    def puck_trace(self) -> np.ndarray:
        """(physics ticks, pucks, 2): puck positions after every physics tick of the last control tick."""
        return self._qpos_trace[:, self._puck_qpos]

    @property
    def paddle_trace(self) -> np.ndarray:
        """(physics ticks, paddles, 2): the same for the paddles."""
        return self._qpos_trace[:, self._paddle_qpos]

    @property
    def state(self) -> State:
        puck_pos = self.data.qpos[self._puck_qpos]
        return State(
            time=float(self.data.time),
            puck_pos=puck_pos,
            puck_vel=self.data.qvel[self._puck_dof],
            paddle_pos=self.data.qpos[self._paddle_qpos],
            paddle_vel=self.data.qvel[self._paddle_dof],
            goals=tuple(self._goal_at(*position) for position in puck_pos),
            touches=self._touches.copy(),
        )

    def _goal_at(self, x: float, y: float) -> Goal | None:
        in_slot = abs(y) < self.geometry.goal_half_width
        if in_slot and x > self.geometry.goal_exit_x:
            return "scored"
        if in_slot and x < -self.geometry.goal_exit_x:
            return "conceded"
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _joint_slots(model: mujoco.MjModel, names_per_thing: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    """Where each thing's (x, y) joints sit in MuJoCo's qpos and qvel: two (things, 2) index arrays."""
    joints = [[model.joint(name) for name in names] for names in names_per_thing]
    qpos = np.array([[int(j.qposadr[0]) for j in row] for row in joints])
    dof = np.array([[int(j.dofadr[0]) for j in row] for row in joints])
    return qpos, dof


def _owner_lookup(model: mujoco.MjModel, geoms_per_thing: list[list[str]]) -> np.ndarray:
    """For every geom id: the index of the puck / paddle it belongs to, or -1."""
    owner = np.full(model.ngeom, -1)
    for index, names in enumerate(geoms_per_thing):
        for name in names:
            owner[model.geom(name).id] = index
    return owner


def _default_puck_positions(count: int) -> np.ndarray:
    """One puck at the centre; more are spread in a row across the centre line."""
    positions = np.zeros((count, 2))
    positions[:, 1] = (np.arange(count) - (count - 1) / 2) * DEFAULT_PUCK_SPACING_M
    return positions


def _rows(values, count: int, default: np.ndarray | None, name: str) -> np.ndarray:
    """`values` as a (count, 2) float array. None gives the default; a single flat (x, y) is one row."""
    if values is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return np.array(default, dtype=float)
    array = np.asarray(values, dtype=float)
    if array.shape == (2,):
        array = array.reshape(1, 2)
    if array.shape != (count, 2):
        raise ValueError(f"{name}: expected {count} row(s) of (x, y), got shape {array.shape}")
    return array
