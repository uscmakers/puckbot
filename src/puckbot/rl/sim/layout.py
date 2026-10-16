"""Layout: what is on the table. How many paddles and pucks, and where each paddle may go.

The physics never changes with the layout: the model builder (model.py) and the env (env.py) just
read it. Add a paddle or a puck here and everything else adapts.

Steps:
    1. Describe  a PaddleSetup per paddle: which end it defends, and its reach
    2. Combine   a Layout = the paddles + the number of pucks

Usage:
    Layout()                    one robot paddle, one puck (the default)
    Layout.of(2)                a robot paddle and an opponent paddle, one puck
    Layout.of(2, pucks=3)
    Layout(paddles=(PaddleSetup("robot", reach="full_table"),))
    Layout(paddles=(PaddleSetup("robot", reach=Box(-0.9, -0.4, -0.3, 0.4)),))

Paddle 0 is the robot's. The robot defends the -x end; an "opponent" defends the +x end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Side = Literal["robot", "opponent"]


@dataclass(frozen=True)
class Box:
    """A rectangle of table positions the paddle centre may use. Metres, CAD frame."""

    x_low: float
    y_low: float
    x_high: float
    y_high: float


Reach = Literal["own_half", "full_table"] | Box
# own_half    the paddle stays on its own side of the centre line (the real robot's case)
# full_table  anywhere on the table, up to the wall margin
# Box         exactly this rectangle (must fit inside the full-table reach)


@dataclass(frozen=True)
class PaddleSetup:
    side: Side = "robot"
    reach: Reach = "own_half"
    home: tuple[float, float] | None = None  # None: the settings' home (mirrored for the opponent)

    @property
    def end_sign(self) -> float:
        """x sign of the end this paddle defends: -1 for the robot, +1 for the opponent."""
        return -1.0 if self.side == "robot" else 1.0


@dataclass(frozen=True)
class Layout:
    paddles: tuple[PaddleSetup, ...] = field(default_factory=lambda: (PaddleSetup(),))
    pucks: int = 1

    def __post_init__(self):
        if not self.paddles:
            raise ValueError("a layout needs at least one paddle")
        if self.pucks < 1:
            raise ValueError("a layout needs at least one puck")

    @classmethod
    def of(cls, paddles: int = 1, pucks: int = 1) -> Layout:
        """The usual setups: paddle 0 is the robot's, paddle 1 the opponent's, then they alternate."""
        sides: tuple[Side, ...] = ("robot", "opponent")
        return cls(tuple(PaddleSetup(sides[i % 2]) for i in range(paddles)), pucks)
