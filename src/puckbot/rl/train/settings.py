"""Settings for training: what the policy sees, the task, and how long to train.

Values live in configs/train.json. This file says what each one means.
Physics numbers (paddle caps, puck friction, ...) are in ../sim/configs/sim.json, not here.
PPO's own hyperparameters are not here either: we use Stable-Baselines3's defaults on purpose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from puckbot.rl.sim.settings import _from_dict

DEFAULT_TRAIN_SETTINGS = Path(__file__).resolve().parent / "configs" / "train.json"


@dataclass
class CameraSettings:
    rate_hz: float  # camera frames per second (the real one: 120)
    latency_s: float  # light on the sensor -> frame available to the policy (a guess until measured)
    puck_noise_m: float  # random error on each reported puck position (Hudson measured 1.5 mm)
    history_lags_frames: tuple[
        int, ...
    ]  # which past frames the policy sees: 0 = newest, 5 = five frames older


@dataclass
class GoalieSettings:
    episode_seconds: float  # an episode ends after this long if nothing happened
    puck_speed_mps: tuple[float, float]  # each serve picks a speed in this range (slow pucks first)
    bank_shot_fraction: float  # share of serves aimed at the goal via a side rail
    blocked_reward: float  # paddle hit the puck and it left our goal
    conceded_reward: float  # puck went into our goal
    approach_reward_per_m: float  # shaping: reward for closing the gap between paddle y and puck y


@dataclass
class GameSettings:
    paddles: int  # 1 = just the robot; 2 = the robot and an opponent (the agent always drives paddle 0)
    pucks: int  # how many pucks are on the table
    opponents: tuple[str, ...]  # scripted skills (sim/skills.py) for the opponent; one per episode
    episode_seconds: float  # an episode ends after this long if nobody scored
    scored_reward: float  # puck into the opponent's goal
    conceded_reward: float  # puck into our goal
    approach_reward_per_m: float  # shaping: reward for closing the distance between our paddle and the puck


@dataclass
class TrainingSettings:
    parallel_envs: int  # simulations running at once (one per CPU core is a good start)
    total_steps: int  # control ticks to train for, summed over all envs
    seed: int


@dataclass
class TrainSettings:
    camera: CameraSettings
    game: GameSettings
    goalie: GoalieSettings
    training: TrainingSettings


def load_train_settings(path: str | Path = DEFAULT_TRAIN_SETTINGS) -> TrainSettings:
    """Read a settings file. A typo'd or missing key raises an error."""
    return _from_dict(TrainSettings, json.loads(Path(path).read_text()))
