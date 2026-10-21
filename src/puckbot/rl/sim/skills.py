"""Scripted skills: hand-written players for one paddle. One implementation for every sim.

A skill controls one paddle (`paddle` = its index), plays puck 0, and ignores any other paddle, so
it works on a table with one paddle or two. It is batched: it computes targets for many worlds at
once (torch), so the same code runs the CPU sim (1 world) and mjlab on a GPU (thousands).

The rules are written for a paddle defending the -x end; for the opponent's paddle the skill mirrors
x on the way in and out, so the same rules serve both ends.

Skills:
    hold            wait at home
    goalie          stand where a moving puck will cross the defence line (side-rail bounces included)
    straight_shot   wind up behind the puck, strike it at the goal centre, go home
    bank_shot       the same, aimed past the nearest side rail so the bounce lands in the goal

Interface (every skill):
    skill = make_skill("bank_shot", table, paddle=0)     # table: the CPU env or mjlab's TableFacts
    skill.reset(env_ids)                                 # new episodes (None = all worlds)
    targets = skill.target(snapshot)                     # (worlds, 2), every control tick
    target = skill.target_one(state)                     # CPU sim: one table's State -> (x, y)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

SKILL_NAMES = ("hold", "goalie", "straight_shot", "bank_shot")

# How much of its speed across the rail the puck keeps when it bounces. Measured in this sim for
# angled hits (30 deg): 0.78. Square-on hits keep less (0.64-0.70). Replace with the real table's
# value once we have recordings.
RAIL_BOUNCE_KEEP = 0.78


# ── What a skill sees ─────────────────────────────────────────────────────────


@dataclass
class Snapshot:
    """Where everything is, for every world."""

    puck_pos: torch.Tensor  # (worlds, pucks, 2) metres
    puck_vel: torch.Tensor
    paddle_pos: torch.Tensor  # (worlds, paddles, 2)
    paddle_vel: torch.Tensor

    @classmethod
    def from_state(cls, state) -> Snapshot:
        """One CPU-sim State (sim/env.py) as a 1-world snapshot."""

        def tensor(array):
            return torch.as_tensor(np.asarray(array)[None], dtype=torch.float32)

        return cls(
            tensor(state.puck_pos), tensor(state.puck_vel), tensor(state.paddle_pos), tensor(state.paddle_vel)
        )


@dataclass
class _View:
    """Puck 0 and this skill's paddle, mirrored so its own goal is at -x. Each (worlds, 2)."""

    puck_pos: torch.Tensor
    puck_vel: torch.Tensor
    paddle_pos: torch.Tensor
    paddle_vel: torch.Tensor


# ── Skills ────────────────────────────────────────────────────────────────────


class Skill:
    """Base class. `table` is anything with geometry, layout, home, paddle_low and paddle_high."""

    def __init__(self, table, paddle: int = 0, device: str = "cpu"):
        self.paddle = paddle
        x_sign = -table.layout.paddles[paddle].end_sign  # +1 robot, -1 opponent: own goal at -x
        self.mirror = torch.tensor([x_sign, 1.0], device=device)
        self.home = (
            torch.as_tensor(np.asarray(table.home[paddle]), dtype=torch.float32, device=device) * self.mirror
        )
        corners = torch.as_tensor(
            np.array([table.paddle_low[paddle], table.paddle_high[paddle]]),
            dtype=torch.float32,
            device=device,
        )
        corners = corners * self.mirror
        self.low, self.high = corners.min(dim=0).values, corners.max(dim=0).values  # its reach, mirrored
        self.geometry = table.geometry

    def reset(self, env_ids=None) -> None:
        """Forget what happened in the last episode (for these worlds; None = all)."""

    def target(self, snapshot: Snapshot) -> torch.Tensor:
        view = _View(
            puck_pos=snapshot.puck_pos[:, 0] * self.mirror,
            puck_vel=snapshot.puck_vel[:, 0] * self.mirror,
            paddle_pos=snapshot.paddle_pos[:, self.paddle] * self.mirror,
            paddle_vel=snapshot.paddle_vel[:, self.paddle] * self.mirror,
        )
        return self._target(view) * self.mirror

    def target_one(self, state) -> np.ndarray:
        """CPU sim: the (x, y) target for one table."""
        return self.target(Snapshot.from_state(state))[0].cpu().numpy().astype(float)

    def _target(self, view: _View) -> torch.Tensor:
        raise NotImplementedError


class Hold(Skill):
    def _target(self, view):
        return self.home.expand_as(view.puck_pos)


class Goalie(Skill):
    """Stand on the defence line where the puck will cross it. Go home if it isn't coming."""

    COMING_SPEED_MPS = 0.05  # slower than this toward our goal counts as "not coming"

    def _target(self, view):
        g = self.geometry
        defence_x = self.home[0]
        reach = g.goal_half_width + g.paddle_radius  # never chase further sideways than this
        speed_toward = -view.puck_vel[:, 0]
        coming = (speed_toward >= self.COMING_SPEED_MPS) & (view.puck_pos[:, 0] >= defence_x)
        seconds = (view.puck_pos[:, 0] - defence_x) / speed_toward.clamp(min=self.COMING_SPEED_MPS)
        crossing = _fold_into_table(view.puck_pos[:, 1] + view.puck_vel[:, 1] * seconds, g.puck_limit_y)
        crossing = crossing.clamp(-reach, reach)
        guard = torch.stack([torch.full_like(crossing, float(defence_x)), crossing], dim=-1)
        return torch.where(coming.unsqueeze(-1), guard, self.home.expand_as(guard))


WINDUP, STRIKE, RECOVER = 0, 1, 2


class Shot(Skill):
    """Windup -> strike -> recover. Each world keeps its own phase.

    Windup:   go to a point behind the puck, on the line from the aim point through the puck
    Strike:   once there and still, drive through the puck toward the aim point
    Recover:  once the puck is clearly moving, go home

    Both the windup point and the strike target are pulled in along the aim line until the paddle can
    reach them: a point outside the paddle's reach would be clipped and bend the shot.
    """

    WINDUP_GAP_M = 0.20  # windup point is at most this far behind the puck centre
    MIN_WINDUP_GAP_M = 0.12  # ...and at least this far (paddle + puck radius is 0.09 m)
    FOLLOW_THROUGH_M = 0.30  # strike target is at most this far past the puck centre
    SETTLED_DISTANCE_M = 0.004  # "at the windup point": a few mm off sideways turns the shot by degrees
    SETTLED_SPEED_MPS = 0.05  # "still"
    HIT_SPEED_MPS = 0.5  # the puck is moving: the shot has been taken

    def __init__(self, table, paddle: int = 0, device: str = "cpu", bank: bool = False):
        super().__init__(table, paddle, device)
        self.bank = bank
        self.phase: torch.Tensor | None = None

    def reset(self, env_ids=None) -> None:
        if self.phase is not None:
            self.phase[slice(None) if env_ids is None else env_ids] = WINDUP

    def _target(self, view):
        worlds = view.puck_pos.shape[0]
        if self.phase is None or self.phase.shape[0] != worlds:
            self.phase = torch.full((worlds,), WINDUP, dtype=torch.long, device=view.puck_pos.device)

        direction = _unit(self._aim(view.puck_pos) - view.puck_pos)
        gap = (
            self._room(view.puck_pos, -direction)
            .clamp(max=self.WINDUP_GAP_M)
            .clamp(min=self.MIN_WINDUP_GAP_M)
        )
        windup_point = view.puck_pos - direction * gap.unsqueeze(-1)
        follow = self._room(view.puck_pos, direction).clamp(min=0.0, max=self.FOLLOW_THROUGH_M)
        strike_point = view.puck_pos + direction * follow.unsqueeze(-1)

        settled = ((view.paddle_pos - windup_point).norm(dim=-1) < self.SETTLED_DISTANCE_M) & (
            view.paddle_vel.norm(dim=-1) < self.SETTLED_SPEED_MPS
        )
        hit = view.puck_vel.norm(dim=-1) > self.HIT_SPEED_MPS
        in_windup, in_strike = self.phase == WINDUP, self.phase == STRIKE

        home = self.home.expand_as(windup_point)
        target = torch.where(
            in_windup.unsqueeze(-1),
            windup_point,
            torch.where((in_strike & ~hit).unsqueeze(-1), strike_point, home),
        )
        self.phase = torch.where(in_windup & settled, STRIKE, self.phase)
        self.phase = torch.where(in_strike & hit, RECOVER, self.phase)
        return target

    def _aim(self, puck_pos: torch.Tensor) -> torch.Tensor:
        """The point to send the puck toward."""
        g = self.geometry
        if not self.bank:  # the goal centre
            return torch.tensor([g.rail_x, 0.0], device=puck_pos.device).expand_as(puck_pos)
        # Past the nearest side rail. A perfect rail would send the puck to the goal's mirror image, 2 x
        # wall away; this rail keeps only RAIL_BOUNCE_KEEP of the across-speed, so aim further out.
        wall_y = torch.where(puck_pos[:, 1] < 0, -g.puck_limit_y, g.puck_limit_y)
        return torch.stack([torch.full_like(wall_y, g.rail_x), wall_y * (1 + 1 / RAIL_BOUNCE_KEEP)], dim=-1)

    def _room(self, start: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """How far the paddle can go from `start` along `direction` and stay inside its reach."""
        edge = torch.where(direction > 0, self.high, self.low)
        steps = torch.where(
            direction.abs() > 1e-9, (edge - start) / direction, torch.full_like(start, float("inf"))
        )
        return steps.min(dim=-1).values.clamp(min=0.0)


# ── Build ─────────────────────────────────────────────────────────────────────


def make_skill(name: str, table, paddle: int = 0, device: str = "cpu") -> Skill:
    """The skill `name` for paddle `paddle`. `table`: the CPU AirHockeyEnv or mjlab's TableFacts."""
    if name == "hold":
        return Hold(table, paddle, device)
    if name == "goalie":
        return Goalie(table, paddle, device)
    if name in ("straight_shot", "bank_shot"):
        return Shot(table, paddle, device, bank=name == "bank_shot")
    raise ValueError(f"unknown skill {name!r}; choose from {SKILL_NAMES}")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _unit(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / vectors.norm(dim=-1, keepdim=True)


def _fold_into_table(y: torch.Tensor, limit: float) -> torch.Tensor:
    """Where a puck heading for `y` really ends up once it bounces off the side rails (mirror bounces)."""
    along = torch.remainder(y + limit, 4 * limit)
    along = torch.where(along > 2 * limit, 4 * limit - along, along)
    return along - limit
