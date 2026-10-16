"""Baselines: how good is a hand-written shot from many different starting spots?

Neither shot looks at a goalie: `straight_shot` aims at the goal centre, `bank_shot` at the goal via
the nearest side rail. With no goalie, the score rate measures AIM only. With the scripted goalie
defending the other end (a two-paddle env), the shortfall is what the goalie blocks.

Steps:
    1. Grid     a resting puck on each point of a grid on our half
    2. Shoot    run the skill from each point (drills.run_skill)
    3. Count    scored / no goal / own goal, plus time and puck speed at the goal
    4. Show     a map of the table (rows = y, columns = x) and a summary line

Usage:
    result = shot_grid(env, "bank_shot")                          # empty goal
    result = shot_grid(AirHockeyEnv(layout=Layout.of(2)), "bank_shot", opponent="goalie")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from puckbot.rl.sim.drills import run_skill
from puckbot.rl.sim.env import AirHockeyEnv

START_X_M = (-0.70, -0.55, -0.40, -0.25)  # wind-up needs 0.2 m of room behind the puck
START_Y_M = tuple(np.linspace(-0.40, 0.40, 7).round(2))
SECONDS = 4.0
MARK = {"scored": "S", "no goal": ".", "conceded": "X"}  # conceded = the puck went in our own goal


@dataclass(frozen=True)
class GridResult:
    skill: str
    opponent: str | None
    outcomes: np.ndarray  # (len(START_Y_M), len(START_X_M)) strings
    seconds: np.ndarray  # time of the last tick
    goal_speed: np.ndarray  # puck speed at the last tick (m/s)

    @property
    def scored(self) -> np.ndarray:
        return self.outcomes == "scored"

    def __str__(self) -> str:
        against = f" vs {self.opponent}" if self.opponent else " (empty goal)"
        lines = [f"{self.skill}{against}   x: " + "  ".join(f"{x:+.2f}" for x in START_X_M)]
        for y, row in zip(START_Y_M, self.outcomes, strict=True):
            lines.append(f"   y {y:+.2f}      " + "      ".join(MARK[outcome] for outcome in row))
        scored = self.scored
        lines.append(
            f"   scored {scored.sum()}/{scored.size}"
            f"   mean time to goal {self.seconds[scored].mean():.2f} s"
            f"   mean puck speed at goal {self.goal_speed[scored].mean():.2f} m/s"
            if scored.any()
            else f"   scored 0/{scored.size}"
        )
        return "\n".join(lines)


def shot_grid(env: AirHockeyEnv, skill: str, opponent: str | None = None) -> GridResult:
    shape = (len(START_Y_M), len(START_X_M))
    outcomes = np.empty(shape, dtype=object)
    seconds, goal_speed = np.zeros(shape), np.zeros(shape)
    for row, y in enumerate(START_Y_M):
        for column, x in enumerate(START_X_M):
            run = run_skill(env, skill, (x, y), (0.0, 0.0), SECONDS, opponent=opponent)
            outcomes[row, column] = run.outcome
            seconds[row, column] = run.time[-1]
            goal_speed[row, column] = np.linalg.norm(run.puck_vel[-1, 0])
    return GridResult(skill, opponent, outcomes, seconds, goal_speed)
