"""Check the mjlab game env with scripted policies for our paddle, per opponent skill.

    hold      our paddle never moves (the floor)
    scripted  our paddle plays the matching skill: goalie against a shooter, straight shot otherwise
Counts scored / conceded / timeout for each opponent skill. Doing nothing should concede to the
shooters; the scripted goalie should block most of their shots.

Usage (mjlab venv):
    python -m puckbot.rl.mjlab.check_game cpu        # 8 worlds, slow but works anywhere
    python -m puckbot.rl.mjlab.check_game cuda:0     # 1024 worlds
"""

from __future__ import annotations

import collections
import sys
import time

import torch
from mjlab.envs import ManagerBasedRlEnv

from puckbot.rl.mjlab.events import SHOOTERS
from puckbot.rl.mjlab.game import game_env_cfg
from puckbot.rl.mjlab.state import TableState
from puckbot.rl.sim.skills import Snapshot, make_skill


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    envs, steps = (8, 150) if device == "cpu" else (1024, 400)
    env = ManagerBasedRlEnv(game_env_cfg(num_envs=envs), device=device)
    paddle = env.action_manager.get_term("paddle")
    facts = paddle.cfg.facts
    has_opponent = "opponent" in env.action_manager.active_terms
    names = env.action_manager.get_term("opponent").skill_names if has_opponent else ("none",)
    ours = {
        "goalie": make_skill("goalie", facts, 0, device),
        "straight_shot": make_skill("straight_shot", facts, 0, device),
    }
    state = TableState(env, facts)
    full_nudge_m = facts.settings.paddle.max_speed_mps * env.step_dt

    for policy in ("hold", "scripted"):
        env.reset()
        for skill in ours.values():
            skill.reset()
        counts = {name: collections.Counter() for name in names}
        started = time.time()
        for _ in range(steps):
            choice = (
                env.action_manager.get_term("opponent").choice
                if has_opponent
                else torch.zeros(envs, dtype=torch.long)
            )
            if policy == "hold":
                action = torch.zeros(envs, 2, device=device)
            else:
                snapshot: Snapshot = state.read()
                defend = torch.tensor(
                    [names[i] in SHOOTERS for i in choice.tolist()], device=device
                ).unsqueeze(-1)
                target = torch.where(
                    defend, ours["goalie"].target(snapshot), ours["straight_shot"].target(snapshot)
                )
                action = ((target - paddle.command_point) / full_nudge_m).clamp(-1, 1)
            opponent_of_world = [names[i] for i in choice.tolist()]
            _, _, terminated, truncated, _ = env.step(action)
            judge = env._game_judge
            done = torch.nonzero(terminated | truncated).flatten()
            for i in done.tolist():
                outcome = "timeout" if truncated[i] else ("scored" if judge.scored[i] else "conceded")
                counts[opponent_of_world[i]][outcome] += 1
            for skill in ours.values():
                skill.reset(done)
        seconds = time.time() - started
        print(f"{policy} ({steps} steps x {envs} worlds, {seconds:.0f} s):")
        for name, count in counts.items():
            print(f"   vs {name:14s} {dict(count)}")


if __name__ == "__main__":
    main()
