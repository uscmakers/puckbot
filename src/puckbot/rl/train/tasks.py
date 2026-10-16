"""Tasks: what the agent is trying to do, as a reset rule and a judge.

The game env (game_env.py) is the same for every task. A task only decides two things:
    reset   where the pucks and paddles start each episode
    judge   after every tick: the reward, and whether the episode has ended and how

So a new task (a shooter, the full game) is a new class here, not a new env.

Usage:
    task = GoalieTask(settings.goalie, sim)
    state, info = task.reset(random, sim)
    verdict = task.judge(state, tick)       # Verdict(reward, outcome, truncated)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from puckbot.rl.sim.env import AirHockeyEnv, State
from puckbot.rl.train.serves import random_serve
from puckbot.rl.train.settings import GoalieSettings

AWAY_SPEED_MPS = 0.05  # moving away from our goal faster than this counts as "moving away"
DISCOUNT = 0.99  # PPO's default discount; the shaping below is potential-based with this discount


@dataclass(frozen=True)
class Verdict:
    reward: float
    outcome: str | None  # None while the episode goes on
    truncated: bool = False  # True: ended by the clock, not by the task's own end


class Task(Protocol):
    max_ticks: int

    def reset(self, random: np.random.Generator, sim: AirHockeyEnv) -> tuple[State, dict]:
        """Place everything for a new episode. Returns the state and any extra info."""

    def judge(self, state: State, tick: int) -> Verdict:
        """Reward and outcome after `tick` control ticks."""


class GoalieTask:
    """Stop random shots at our goal (serves.py).

    Reward: +1 blocked, -1 conceded, plus a small shaping term: each tick, the shrinking of the gap
    between paddle y and puck y (approach_reward_per_m x metres). A gap closed and then reopened nets
    to zero, so standing near the puck earns nothing by itself (potential-based shaping).

    Ends:
        blocked   our paddle touched the puck and it is now moving away from our goal   (+1)
        conceded  the puck went into our goal                                           (-1)
        missed    the puck is moving away without being touched (it hit the end rail)    (0)
        timeout   nothing happened for episode_seconds                                   (0)
    """

    def __init__(self, goalie: GoalieSettings, sim: AirHockeyEnv):
        if sim.n_pucks != 1:
            raise ValueError("GoalieTask serves one puck")
        self.goalie = goalie
        self.max_ticks = round(goalie.episode_seconds / sim.step_seconds)

    def reset(self, random, sim):
        serve = random_serve(random, sim.geometry, self.goalie)
        state = sim.reset(puck_pos=serve.puck_pos, puck_vel=serve.puck_vel)
        self.touched = False
        self.gap = self._gap_m(state)
        return state, {"serve_banks": serve.banks}

    def judge(self, state, tick):
        self.touched = self.touched or bool(state.touches[0].any())
        outcome = self._outcome(state, tick)
        reward = {"blocked": self.goalie.blocked_reward, "conceded": self.goalie.conceded_reward}.get(
            outcome, 0.0
        )

        gap = self._gap_m(state)
        reward += self.goalie.approach_reward_per_m * (self.gap - DISCOUNT * gap)
        self.gap = gap
        return Verdict(reward, outcome, truncated=outcome == "timeout")

    def _outcome(self, state: State, tick: int) -> str | None:
        if state.goal == "conceded":
            return "conceded"
        if state.puck_vel[0, 0] > AWAY_SPEED_MPS:
            return "blocked" if self.touched else "missed"
        return "timeout" if tick >= self.max_ticks else None

    @staticmethod
    def _gap_m(state: State) -> float:
        return float(abs(state.paddle_pos[0, 1] - state.puck_pos[0, 1]))
