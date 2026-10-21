"""Build the MuJoCo model: the CAD table with flat-sliding pucks and paddles.

Steps:
    1. Load      the CAD model (cad/table_urdf/table.xml)
    2. Clone     the CAD puck and paddle into as many as the Layout asks for (puck_0, paddle_0, ...)
    3. Flatten   they slide on the surface instead of being free 3D bodies; each paddle joint gets a
                 MuJoCo position motor (stiffness, damping, force cap), like a stepper holding its position
    4. Contacts  they ignore the deck (air cushion) but hit the rails and each other
    5. Measure   read the table's size back from the model (TableGeometry)

Usage:
    model = build_model(load_settings())                 # one robot paddle, one puck
    spec = build_spec(load_settings())                   # the same, not compiled (for mjlab)
    model = build_model(load_settings(), Layout.of(2))   # + an opponent paddle
    geometry = TableGeometry.from_model(model)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco

from puckbot.rl.sim.layout import Box, Layout, PaddleSetup
from puckbot.rl.sim.settings import DEFAULT_TABLE_MODEL, PaddleSettings, Settings

# Two geoms collide when one's `contype` matches the other's `conaffinity` (the CAD geoms use 1).
DECK_GROUP = 2  # the deck: nothing slides on it, the air cushion carries the puck

VIDEO_WIDTH_PX, VIDEO_HEIGHT_PX = 1280, 720  # biggest video frame the model can render
OPPONENT_COLOUR = [0.30, 0.50, 0.85, 1.0]  # so the two paddles can be told apart in videos

# Names inside the model. Everything that looks things up by name uses these.


def puck_geom(index: int) -> str:
    return f"puck_{index}_col"


def paddle_geom(index: int, part: str = "base") -> str:
    return f"paddle_{index}_{part}_col"


def puck_joints(index: int) -> list[str]:
    return [f"puck_{index}_x", f"puck_{index}_y"]


def paddle_joints(index: int) -> list[str]:
    return [f"paddle_{index}_x", f"paddle_{index}_y"]


# ── Table size ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TableGeometry:
    """Sizes read from the CAD model. Metres."""

    rail_x: float  # inner face of each end rail (the goal line)
    rail_y: float  # inner face of each side rail
    goal_half_width: float  # goal slot is |y| < this
    goal_exit_x: float  # a puck past this |x| (inside the slot) has left the table
    puck_radius: float
    paddle_radius: float

    @classmethod
    def from_model(
        cls, model: mujoco.MjModel, puck: str = puck_geom(0), paddle: str = paddle_geom(0)
    ) -> TableGeometry:
        side_rail = model.geom("rail_pos_y")
        end_rail = model.geom("endrail_pos_x_pos_y")
        lintel = model.geom("goal_lintel_pos_x")
        rim = model.geom("rim_pos_x")
        return cls(
            rail_x=float(end_rail.pos[0] - end_rail.size[0]),
            rail_y=float(side_rail.pos[1] - side_rail.size[1]),
            goal_half_width=float(lintel.size[1]),
            goal_exit_x=float(rim.pos[0] - rim.size[0]),
            puck_radius=float(model.geom(puck).size[0]),
            paddle_radius=float(model.geom(paddle).size[0]),
        )

    @property
    def puck_limit_y(self) -> float:
        """Furthest y a puck's centre can be before it touches a side rail."""
        return self.rail_y - self.puck_radius


# ── Build ─────────────────────────────────────────────────────────────────────


def build_model(
    settings: Settings, layout: Layout | None = None, table_model: str | Path = DEFAULT_TABLE_MODEL
) -> mujoco.MjModel:
    return build_spec(settings, layout, table_model).compile()


def build_spec(
    settings: Settings, layout: Layout | None = None, table_model: str | Path = DEFAULT_TABLE_MODEL
) -> mujoco.MjSpec:
    """The model before compiling (mjlab builds its scene from this)."""
    layout = layout or Layout()
    spec = mujoco.MjSpec.from_file(str(table_model))
    # The CAD file's own puck and paddle are the templates: read the table size from them first.
    geometry = TableGeometry.from_model(spec.compile(), puck="puck_col", paddle="paddle_base_col")

    spec.option.timestep = settings.physics_timestep_s
    spec.visual.global_.offwidth = VIDEO_WIDTH_PX  # offscreen buffer, so video.py can render this big
    spec.visual.global_.offheight = VIDEO_HEIGHT_PX
    spec.visual.headlight.specular = [0, 0, 0]  # no glare: a top-down camera washes out the puck without this
    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    _let_things_slide_over_the_deck(spec)

    puck_template, paddle_template = spec.body("puck"), spec.body("paddle")
    for index in range(layout.pucks):
        _add_puck(spec, puck_template, index, settings)
    for index, setup in enumerate(layout.paddles):
        low, high = paddle_limits(setup, settings, geometry)
        _add_paddle(spec, paddle_template, index, setup, low, high, settings)
    spec.delete(puck_template)
    spec.delete(paddle_template)
    return spec


def _let_things_slide_over_the_deck(spec: mujoco.MjSpec) -> None:
    deck = spec.geom("deck_apron")
    deck.contype = DECK_GROUP
    deck.conaffinity = DECK_GROUP


def _clone_body(spec: mujoco.MjSpec, template, name: str, rename_geom, colour=None):
    """A copy of the template body (inertia and shapes) under a new name, with no joint yet."""
    body = spec.worldbody.add_body(name=name, pos=template.pos)
    body.explicitinertial = template.explicitinertial
    body.mass, body.inertia = template.mass, template.inertia
    body.ipos, body.iquat = template.ipos, template.iquat
    for geom in template.geoms:
        copy = body.add_geom(
            name=rename_geom(geom.name) if geom.name else "",  # named at creation: older MuJoCo needs it
            type=geom.type,
            size=geom.size,
            pos=geom.pos,
            quat=geom.quat,
            rgba=geom.rgba,
            friction=geom.friction,
            solref=geom.solref,
            solimp=geom.solimp,
            priority=geom.priority,
            contype=geom.contype,
            conaffinity=geom.conaffinity,
            group=geom.group,
            density=geom.density,
            meshname=geom.meshname,
        )
        if not geom.name and colour is not None:  # the unnamed geom is the picture; named ones are physics
            copy.rgba = colour
    body.pos[:2] = 0  # so the joint values are table positions
    return body


def _add_puck(spec: mujoco.MjSpec, template, index: int, settings: Settings) -> None:
    puck = _clone_body(spec, template, f"puck_{index}", lambda name: name.replace("puck", f"puck_{index}", 1))
    for name, axis in zip(puck_joints(index), [(1, 0, 0), (0, 1, 0)], strict=True):
        puck.add_joint(
            name=name,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=axis,
            damping=settings.puck.slowdown_ns_per_m,
        )
    puck.add_joint(name=f"puck_{index}_yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=(0, 0, 1), damping=1e-5)
    # by handle, not spec.geom(name): older MuJoCo can't look up names added after a compile
    collision = next(geom for geom in puck.geoms if geom.name == puck_geom(index))
    collision.solref = [0.004, settings.puck.bounce_damping_ratio]


def _add_paddle(
    spec: mujoco.MjSpec, template, index: int, setup: PaddleSetup, low, high, settings: Settings
) -> None:
    colour = OPPONENT_COLOUR if setup.side == "opponent" else None
    paddle = _clone_body(
        spec, template, f"paddle_{index}", lambda name: name.replace("paddle", f"paddle_{index}", 1), colour
    )
    for axis_index, name in enumerate(paddle_joints(index)):
        axis = [0, 0, 0]
        axis[axis_index] = 1
        paddle.add_joint(
            name=name,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=axis,
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            range=[low[axis_index], high[axis_index]],
        )
        _add_position_motor(spec, name, settings.paddle, template.mass)


def _add_position_motor(spec: mujoco.MjSpec, joint: str, paddle: PaddleSettings, paddle_mass: float) -> None:
    """MuJoCo's own position actuator: force = kp (target - position) - kv velocity.

    A critically damped spring that pulls the paddle to its target: it catches up in about
    `response_time_s` and does not overshoot. kp = m w^2 and kv = 2 m w with w = 1 / response time.
    One actuator per joint, in joint order. No force cap: the paddle is just a mass on a spring.
    """
    omega = 1.0 / paddle.response_time_s
    stiffness, damping = paddle_mass * omega**2, 2 * paddle_mass * omega
    spec.add_actuator(
        name=f"{joint}_motor",
        target=joint,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=[stiffness] + [0.0] * 9,
        biasprm=[0.0, -stiffness, -damping] + [0.0] * 7,
    )


# ── Where a paddle may go ─────────────────────────────────────────────────────


def paddle_limits(
    setup: PaddleSetup, settings: Settings, geometry: TableGeometry
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Where the paddle centre may go: ((x_low, y_low), (x_high, y_high)).

    It always keeps `wall_margin_m` off the rails. Beyond that, `setup.reach` decides.
    """
    margin = settings.paddle.wall_margin_m + geometry.paddle_radius
    wall_x, wall_y = geometry.rail_x - margin, geometry.rail_y - margin
    reach = setup.reach

    if reach == "full_table":
        return (-wall_x, -wall_y), (wall_x, wall_y)
    if reach == "own_half":
        # forward: reach_from_end_m from its own end rail, but never closer than a radius to the centre line
        forward = max(geometry.rail_x - settings.paddle.reach_from_end_m, geometry.paddle_radius)
        if setup.end_sign < 0:
            return (-wall_x, -wall_y), (-forward, wall_y)
        return (forward, -wall_y), (wall_x, wall_y)
    if isinstance(reach, Box):
        inside = (
            -wall_x <= reach.x_low < reach.x_high <= wall_x
            and -wall_y <= reach.y_low < reach.y_high <= wall_y
        )
        if not inside:
            raise ValueError(
                f"{reach} must fit inside the table's full reach x +-{wall_x:.3f}, y +-{wall_y:.3f}"
            )
        return (reach.x_low, reach.y_low), (reach.x_high, reach.y_high)
    raise ValueError(f"unknown reach {reach!r}")


def paddle_home(
    setup: PaddleSetup, settings: Settings, limits: tuple[tuple[float, float], tuple[float, float]]
) -> tuple[float, float]:
    """Where the paddle waits: its own home if given, else the settings' home (mirrored for the opponent).

    Always pulled inside the paddle's limits, so a small reach box can never start the paddle outside it.
    """
    home_x, home_y = setup.home or (
        settings.paddle.home_position_m[0] * -setup.end_sign,
        settings.paddle.home_position_m[1],
    )
    (low_x, low_y), (high_x, high_y) = limits
    return min(max(home_x, low_x), high_x), min(max(home_y, low_y), high_y)
