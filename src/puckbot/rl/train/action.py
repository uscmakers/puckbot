"""Action: what the policy outputs, and what the paddle does with it.

The policy outputs (dx, dy) in [-1, 1]. They nudge the commanded point:
    +-1 = one control tick of travel at the paddle's top speed (3 m/s x 20 ms = 6 cm)
     0 = hold still
The sim's top speed still applies, so the nudge can never ask for the impossible.

Usage:
    target = action_to_target(action, env)         # (x, y); then env.step(...)
    action = target_to_action(target, env)         # the other way (to run a hand-written skill)
Both take `paddle=` (default 0, the robot's).
"""

from __future__ import annotations

import numpy as np

from puckbot.rl.sim.env import AirHockeyEnv


def action_to_target(action, env: AirHockeyEnv, paddle: int = 0) -> np.ndarray:
    return env.paddle_target[paddle] + np.clip(action, -1, 1) * _full_nudge_m(env, paddle)


def target_to_action(target, env: AirHockeyEnv, paddle: int = 0) -> np.ndarray:
    return np.clip((np.asarray(target) - env.paddle_target[paddle]) / _full_nudge_m(env, paddle), -1, 1)


def _full_nudge_m(env: AirHockeyEnv, paddle: int) -> float:
    return env.settings.paddle.max_speed_mps * env.step_seconds
