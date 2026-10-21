"""Drills: run one skill from a fixed starting setup and record what happens.

Steps:
    1. Start    put the puck (and paddles) at the starting spot
    2. Loop     every control tick: each skill picks its paddle's target -> env steps
    3. Stop     on a goal, or when time is up
    4. Record   keep every tick so the run can be saved and replayed

Usage:
    env = AirHockeyEnv()
    run = run_drill(env, "bank_shot")
    run.outcome         # "scored", "conceded" or "no goal"
    run = run_skill(env2, "straight_shot", (-0.4, 0.1), (0, 0), 4.0, opponent="goalie")  # two paddles
    run.save("run.npz")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from puckbot.rl.sim.env import AirHockeyEnv
from puckbot.rl.sim.skills import make_skill

# ── Drills ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Drill:
    skill: str
    puck_pos: tuple[float, float]
    puck_vel: tuple[float, float]
    seconds: float
    about: str


DRILLS = {
    "hold": Drill("hold", (0.0, 0.0), (0.0, 0.0), 2.0, "paddle waits at home, puck sits at centre ice"),
    "straight_shot": Drill("straight_shot", (-0.40, 0.15), (0.0, 0.0), 4.0, "hit a resting puck at the goal"),
    "bank_shot": Drill(
        "bank_shot", (-0.40, 0.15), (0.0, 0.0), 4.0, "hit a resting puck at the goal via a side rail"
    ),
    "goalie": Drill("goalie", (0.60, 0.25), (-2.0, -0.35), 3.0, "block a shot coming at our goal"),
}


# ── Result ────────────────────────────────────────────────────────────────────


@dataclass
class Run:
    drill: str
    outcome: str  # "scored", "conceded" or "no goal"
    time: np.ndarray  # (n,) seconds
    qpos: np.ndarray  # (n, nq) every joint: enough to replay the run
    puck_pos: np.ndarray  # (n, pucks, 2) metres
    puck_vel: np.ndarray
    paddle_pos: np.ndarray  # (n, paddles, 2)
    paddle_vel: np.ndarray
    paddle_target: np.ndarray  # (n, paddles, 2) the commanded points the paddles were pulled toward

    @property
    def top_puck_speed(self) -> float:
        return float(np.linalg.norm(self.puck_vel, axis=-1).max())

    def save(self, path: str | Path) -> None:
        arrays = {k: v for k, v in self.__dict__.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(path, drill=self.drill, outcome=self.outcome, **arrays)


# ── Run ───────────────────────────────────────────────────────────────────────


def run_drill(env: AirHockeyEnv, name: str) -> Run:
    drill = DRILLS[name]
    return run_skill(env, drill.skill, drill.puck_pos, drill.puck_vel, drill.seconds, label=name)


def run_skill(
    env: AirHockeyEnv,
    skill_name: str,
    puck_pos,
    puck_vel,
    seconds: float,
    label: str | None = None,
    opponent: str | None = None,
) -> Run:
    """Run a skill on paddle 0 from any starting puck, and record every tick.

    `opponent` is a skill for paddle 1 (the env needs two paddles). Paddles with no skill wait at home.
    """
    skills = [make_skill(skill_name, env)]
    if opponent:
        skills.append(make_skill(opponent, env, paddle=1))
    skills += [make_skill("hold", env, paddle=k) for k in range(len(skills), env.n_paddles)]  # the rest wait
    for skill in skills:
        skill.reset()

    state = env.reset(puck_pos=puck_pos, puck_vel=puck_vel)
    ticks = [(state, env.data.qpos.copy(), env.paddle_target)]
    for _ in range(round(seconds / env.step_seconds)):
        if state.goal:
            break
        state = env.step(np.array([skill.target_one(state) for skill in skills]))
        ticks.append((state, env.data.qpos.copy(), env.paddle_target))

    states, qpos, targets = zip(*ticks, strict=True)
    return Run(
        drill=label or skill_name,
        outcome=state.goal or "no goal",
        time=np.array([s.time for s in states]),
        qpos=np.array(qpos),
        puck_pos=np.array([s.puck_pos for s in states]),
        puck_vel=np.array([s.puck_vel for s in states]),
        paddle_pos=np.array([s.paddle_pos for s in states]),
        paddle_vel=np.array([s.paddle_vel for s in states]),
        paddle_target=np.array(targets),
    )
