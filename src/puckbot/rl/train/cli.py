"""Command line: train the goalie, or judge a model or a hand-written skill.

Usage:
    puckbot-train train                                  PPO with train.json (writes outputs/train/)
    puckbot-train train --steps 200000 --envs 4          shorter run, 4 processes
    puckbot-train evaluate --model outputs/train/goalie_ppo.zip
    puckbot-train evaluate --skill goalie                the hand-written goalie, for comparison
    puckbot-train evaluate --skill hold                  doing nothing, the floor
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from puckbot.rl.train.settings import DEFAULT_TRAIN_SETTINGS, load_train_settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--settings", default=str(DEFAULT_TRAIN_SETTINGS), help="settings file (JSON)")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train the goalie with PPO")
    train.add_argument("--steps", type=int, help="total control ticks (default: from settings)")
    train.add_argument("--envs", type=int, help="parallel processes (default: from settings)")
    train.add_argument("--out-dir", default="outputs/train")

    judge = commands.add_parser("evaluate", help="play fixed serves and count outcomes")
    source = judge.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="a saved PPO model (.zip)")
    source.add_argument("--skill", help="a hand-written skill: hold, goalie")
    judge.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args(argv)

    settings = load_train_settings(args.settings)

    if args.command == "train":
        from puckbot.rl.train.train import train as run_training  # heavy imports only when needed

        overrides = {"total_steps": args.steps, "parallel_envs": args.envs}
        overrides = {key: value for key, value in overrides.items() if value is not None}
        run_training(replace(settings, training=replace(settings.training, **overrides)), args.out_dir)
        return

    from puckbot.rl.train.evaluate import evaluate, model_policy, skill_policy
    from puckbot.rl.train.game_env import GameEnv

    if args.model:
        from stable_baselines3 import PPO

        policy = model_policy(PPO.load(args.model))
    else:
        policy = skill_policy(args.skill)
    print(evaluate(policy, episodes=args.episodes, env=GameEnv(settings)))


if __name__ == "__main__":
    main()
