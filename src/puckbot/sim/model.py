"""Build the MuJoCo model: the CAD table with a flat-sliding puck and paddle.

Steps:
    1. Load      the CAD model (cad/table_urdf/table.xml)
    2. Flatten   puck and paddle slide on the surface instead of being free 3D bodies
    3. Contacts  puck and paddle ignore the deck (air cushion) but hit rails and each other
    4. Joints    the paddle gets slide joints (x, y) with limits; the puck gets a slow-down
    5. Measure   read the table's size back from the model (TableGeometry)

Usage:
    model = build_model(load_settings())
    geometry = TableGeometry.from_model(model)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco

from puckbot.sim.settings import DEFAULT_TABLE_MODEL, Settings

# Two geoms collide when one's `contype` matches the other's `conaffinity` (the CAD geoms use 1).
DECK_GROUP = 2  # the deck: nothing slides on it, the air cushion carries the puck

VIDEO_WIDTH_PX, VIDEO_HEIGHT_PX = 1280, 720  # biggest video frame the model can render


# ── Table size ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TableGeometry:
    """Sizes read from the CAD model. Metres."""

    rail_x: float  # inner face of each end rail (the goal line)
    rail_y: float  # inner face of each side rail
    goal_half_width: float  # goal slot is |y| < this
    goal_exit_x: float  # a puck past this x (inside the slot) has left the table
    puck_radius: float
    paddle_radius: float

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> TableGeometry:
        side_rail = model.geom("rail_pos_y")
        end_rail = model.geom("endrail_pos_x_pos_y")
        lintel = model.geom("goal_lintel_pos_x")
        rim = model.geom("rim_pos_x")
        return cls(
            rail_x=float(end_rail.pos[0] - end_rail.size[0]),
            rail_y=float(side_rail.pos[1] - side_rail.size[1]),
            goal_half_width=float(lintel.size[1]),
            goal_exit_x=float(rim.pos[0] - rim.size[0]),
            puck_radius=float(model.geom("puck_col").size[0]),
            paddle_radius=float(model.geom("paddle_base_col").size[0]),
        )

    @property
    def puck_limit_y(self) -> float:
        """Furthest y a puck's centre can be before it touches a side rail."""
        return self.rail_y - self.puck_radius


# ── Build ─────────────────────────────────────────────────────────────────────


def build_model(settings: Settings, table_model: str | Path = DEFAULT_TABLE_MODEL) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(table_model))
    geometry = TableGeometry.from_model(spec.compile())

    spec.option.timestep = settings.physics_timestep_s
    spec.visual.global_.offwidth = VIDEO_WIDTH_PX  # offscreen buffer, so video.py can render this big
    spec.visual.global_.offheight = VIDEO_HEIGHT_PX
    spec.visual.headlight.specular = [0, 0, 0]  # no glare: a top-down camera washes out the puck without this
    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    _let_things_slide_over_the_deck(spec)
    _flatten_puck(spec, settings)
    _flatten_paddle(spec, settings, geometry)
    return spec.compile()


def _let_things_slide_over_the_deck(spec: mujoco.MjSpec) -> None:
    deck = spec.geom("deck_apron")
    deck.contype = DECK_GROUP
    deck.conaffinity = DECK_GROUP


def _flatten_puck(spec: mujoco.MjSpec, settings: Settings) -> None:
    puck = spec.body("puck")
    spec.delete(spec.joint("puck_free"))
    for name, axis in [("puck_x", (1, 0, 0)), ("puck_y", (0, 1, 0))]:
        puck.add_joint(
            name=name,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=axis,
            damping=settings.puck.slowdown_ns_per_m,
        )
    puck.add_joint(name="puck_yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=(0, 0, 1), damping=1e-5)
    spec.geom("puck_col").solref = [0.004, settings.puck.bounce_damping_ratio]


def _flatten_paddle(spec: mujoco.MjSpec, settings: Settings, geometry: TableGeometry) -> None:
    paddle = spec.body("paddle")
    spec.delete(spec.joint("paddle_free"))
    paddle.pos[:2] = 0  # so the joint values are table positions

    low, high = paddle_limits(settings, geometry)
    for axis_index, name in enumerate(["paddle_x", "paddle_y"]):
        axis = [0, 0, 0]
        axis[axis_index] = 1
        paddle.add_joint(
            name=name,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=axis,
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            range=[low[axis_index], high[axis_index]],
        )


def paddle_limits(
    settings: Settings, geometry: TableGeometry
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Where the paddle centre may go: ((x_low, y_low), (x_high, y_high)).

    It keeps `wall_margin_m` off the rails and stays on the robot's half (x < 0).
    """
    margin = settings.paddle.wall_margin_m + geometry.paddle_radius
    low = (-(geometry.rail_x - margin), -(geometry.rail_y - margin))
    high = (-geometry.paddle_radius, geometry.rail_y - margin)
    return low, high
