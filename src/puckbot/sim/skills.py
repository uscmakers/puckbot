"""Hand-written skills: simple rules that pick the paddle's next target.

A skill looks at the State and returns the (x, y) the paddle should head for.
No learning: these are the baseline a trained policy has to beat.

Skills:
    Hold         wait at the home position
    Shot         wind up behind the puck, then hit it toward an aim point
                   straight_shot  aims at the goal
                   bank_shot      aims past the nearest side rail so the bounce lands in the goal
    Goalie       stand where a moving puck will cross our defence line (side-rail bounces included)

Usage:
    skill = make_skill("bank_shot", env)
    skill.reset()
    target = skill.target(state)        # every control tick
"""

from __future__ import annotations

import numpy as np

from puckbot.sim.env import AirHockeyEnv, State

SKILL_NAMES = ("hold", "straight_shot", "bank_shot", "goalie")

# How much of its speed across the rail the puck keeps when it bounces. Measured in this sim for
# angled hits (30 deg): 0.78. Square-on hits keep less (0.64-0.70). Replace with the real table's
# value once we have recordings.
RAIL_BOUNCE_KEEP = 0.78


# ── Skills ────────────────────────────────────────────────────────────────────


class Skill:
    def reset(self) -> None:
        """Forget what happened in the last run."""

    def target(self, state: State) -> np.ndarray:
        raise NotImplementedError


class Hold(Skill):
    def __init__(self, home):
        self.home = np.asarray(home)

    def target(self, state: State) -> np.ndarray:
        return self.home


class Shot(Skill):
    """Windup -> strike -> recover.

    Windup:   go to a point behind the puck, on the line from the aim point through the puck
    Strike:   once there and still, drive through the puck toward the aim point
    Recover:  once the puck is clearly moving, go home
    """

    WINDUP_GAP_M = 0.20  # windup point is this far behind the puck centre
    FOLLOW_THROUGH_M = 0.30  # strike target is this far past the puck centre
    SETTLED_DISTANCE_M = 0.02  # "at the windup point"
    SETTLED_SPEED_MPS = 0.10  # "still"
    HIT_SPEED_MPS = 0.5  # the puck is moving: the shot has been taken

    def __init__(self, aim_at, home):
        self.aim_at = aim_at  # function: puck position -> point to send the puck toward
        self.home = np.asarray(home)
        self.reset()

    def reset(self) -> None:
        self.phase = "windup"

    def target(self, state: State) -> np.ndarray:
        direction = _unit(self.aim_at(state.puck_pos) - state.puck_pos)
        windup_point = state.puck_pos - direction * self.WINDUP_GAP_M

        if self.phase == "windup":
            settled = (
                np.linalg.norm(state.paddle_pos - windup_point) < self.SETTLED_DISTANCE_M
                and np.linalg.norm(state.paddle_vel) < self.SETTLED_SPEED_MPS
            )
            if settled:
                self.phase = "strike"
            return windup_point

        if self.phase == "strike":
            if np.linalg.norm(state.puck_vel) > self.HIT_SPEED_MPS:
                self.phase = "recover"
                return self.home
            return state.puck_pos + direction * self.FOLLOW_THROUGH_M

        return self.home


class Goalie(Skill):
    """Stand on the defence line where the puck will cross it. Go home if it isn't coming."""

    COMING_SPEED_MPS = 0.05  # slower than this toward our goal counts as "not coming"

    def __init__(self, geometry, home, goal_reach_m: float):
        self.home = np.asarray(home)
        self.defence_x = self.home[0]
        self.puck_limit_y = geometry.puck_limit_y
        self.goal_reach_m = goal_reach_m  # never chase further sideways than this

    def target(self, state: State) -> np.ndarray:
        puck_x, puck_y = state.puck_pos
        speed_toward_us = -state.puck_vel[0]
        if speed_toward_us < self.COMING_SPEED_MPS or puck_x < self.defence_x:
            return self.home

        seconds_to_line = (puck_x - self.defence_x) / speed_toward_us
        crossing_y = _fold_into_table(puck_y + state.puck_vel[1] * seconds_to_line, self.puck_limit_y)
        return np.array([self.defence_x, np.clip(crossing_y, -self.goal_reach_m, self.goal_reach_m)])


# ── Build ─────────────────────────────────────────────────────────────────────


def make_skill(name: str, env: AirHockeyEnv) -> Skill:
    geometry, home = env.geometry, env.home

    def goal_centre(_puck_pos):
        return np.array([geometry.rail_x, 0.0])

    def past_the_rail(puck_pos):
        """Aim past the nearest side rail so the bounce lands in the goal.

        A perfect rail would send the puck to the goal's mirror image, 2 x wall away. This rail
        keeps only RAIL_BOUNCE_KEEP of the across-speed, so the puck must be aimed further out.
        """
        wall_y = np.sign(puck_pos[1] or 1.0) * geometry.puck_limit_y
        return np.array([geometry.rail_x, wall_y * (1 + 1 / RAIL_BOUNCE_KEEP)])

    if name == "hold":
        return Hold(home)
    if name == "straight_shot":
        return Shot(goal_centre, home)
    if name == "bank_shot":
        return Shot(past_the_rail, home)
    if name == "goalie":
        reach = geometry.goal_half_width + geometry.paddle_radius
        return Goalie(geometry, home, goal_reach_m=reach)
    raise ValueError(f"unknown skill {name!r}; choose from {SKILL_NAMES}")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _fold_into_table(y: float, limit: float) -> float:
    """Where a puck heading for `y` really ends up once it bounces off the side rails (mirror bounces)."""
    along = (y + limit) % (4 * limit)
    if along > 2 * limit:
        along = 4 * limit - along
    return along - limit
