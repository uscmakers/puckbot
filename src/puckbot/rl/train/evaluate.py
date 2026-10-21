"""Evaluate: judge a policy on a fixed set of serves, not on training reward.

Every policy faces the same starts (episode i always uses seed + i), so scores are comparable:
a trained model, the hand-written goalie, or doing nothing.

Steps:
    1. Start    reset the env with a fixed seed
    2. Play     the policy picks an action every tick until the episode ends
    3. Count    how often each outcome the task reports happened (goalie: blocked, conceded, ...)

Usage:
    report = evaluate(model_policy(model), episodes=200)
    report = evaluate(skill_policy("goalie"))          # the hand-written baseline
    print(report)
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from puckbot.rl.sim.skills import make_skill
from puckbot.rl.train.action import target_to_action
from puckbot.rl.train.game_env import GameEnv

# A policy gets the observation and the env (hand-written skills read the true state from it).
Policy = Callable[[np.ndarray, GameEnv], np.ndarray]


@dataclass(frozen=True)
class Report:
    counts: dict[str, int]

    @property
    def episodes(self) -> int:
        return sum(self.counts.values())

    @property
    def block_rate(self) -> float:
        """Goalie task: of the shots that would have scored, how many were stopped."""
        blocked, conceded = self.counts.get("blocked", 0), self.counts.get("conceded", 0)
        return blocked / (blocked + conceded) if blocked + conceded else float("nan")

    def __str__(self) -> str:
        parts = ", ".join(f"{name} {count}" for name, count in sorted(self.counts.items()))
        return f"{self.episodes} episodes: {parts}; block rate {self.block_rate:.0%}"


def evaluate(policy: Policy, episodes: int = 200, seed: int = 10_000, env: GameEnv | None = None) -> Report:
    env = env or GameEnv()
    counts = Counter()
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        if hasattr(policy, "reset"):
            policy.reset()
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs, env))
            done = terminated or truncated
        counts[info["outcome"]] += 1
    return Report(dict(counts))


# ── Policies ──────────────────────────────────────────────────────────────────


def model_policy(model) -> Policy:
    """A trained Stable-Baselines3 model, acting without exploration noise."""
    return lambda obs, _env: model.predict(obs, deterministic=True)[0]


class SkillPolicy:
    """A hand-written skill as a policy: its target, as an action."""

    def __init__(self, name: str):
        self.skill = make_skill(name, GameEnv().sim)

    def reset(self) -> None:
        self.skill.reset()

    def __call__(self, _obs: np.ndarray, env: GameEnv) -> np.ndarray:
        return target_to_action(self.skill.target_one(env.state), env.sim)


def skill_policy(name: str) -> SkillPolicy:
    """A hand-written skill from sim/skills.py. `hold` = do nothing; `goalie` = the scripted baseline."""
    return SkillPolicy(name)
