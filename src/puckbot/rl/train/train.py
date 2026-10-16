"""Train the goalie with PPO (Stable-Baselines3, all hyperparameters at their defaults).

Steps:
    1. Build     several GameEnv copies, one process each
    2. Train     PPO("MlpPolicy", ...): two hidden layers of 64, SB3's defaults for everything else
    3. Save      checkpoints while training, the final model at the end
    4. Judge     play the fixed serves (evaluate.py) and print the result

Only the number of envs, the number of steps and the seed are ours. Everything else is the library's
default on purpose: no tuning until something is clearly wrong.

Usage:
    model = train(load_train_settings(), out_dir="outputs/train")
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from puckbot.rl.train.evaluate import evaluate, model_policy
from puckbot.rl.train.game_env import GameEnv
from puckbot.rl.train.settings import TrainSettings

CHECKPOINTS = 10  # how many times to save while training


def train(settings: TrainSettings, out_dir: str | Path) -> PPO:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = settings.training

    envs = make_vec_env(
        lambda: GameEnv(settings),
        n_envs=run.parallel_envs,
        seed=run.seed,
        vec_env_cls=SubprocVecEnv if run.parallel_envs > 1 else DummyVecEnv,
    )
    model = PPO("MlpPolicy", envs, seed=run.seed, verbose=1)
    saver = CheckpointCallback(
        save_freq=max(run.total_steps // CHECKPOINTS // run.parallel_envs, 1),
        save_path=str(out_dir),
        name_prefix="goalie_ppo",
    )
    model.learn(total_timesteps=run.total_steps, callback=saver)
    model.save(out_dir / "goalie_ppo")
    envs.close()

    print(evaluate(model_policy(model), env=GameEnv(settings)))
    return model
