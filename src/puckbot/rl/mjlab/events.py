"""Reset events: how each episode starts.

    start_episode   pick the opponent's skill for each world, and place the puck to match:
                      a shooter (straight_shot, bank_shot) gets the puck at rest on its half: we defend
                      the goalie, or no opponent, leaves the puck at rest on our half: we attack

Usage (in the env config, after mjlab's reset_scene_to_default):
    EventTermCfg(func=start_episode, mode="reset", params={"facts": facts})
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from puckbot.rl.mjlab.scene import ENTITY, TableFacts

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

SHOOTERS = ("straight_shot", "bank_shot")
START_X_FRACTION = (0.25, 0.75)  # the puck starts this far from the centre line (fraction of half the length)
START_Y_FRACTION = 0.8  # and within this fraction of the side-rail limit


def start_episode(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, facts: TableFacts) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n, device, g = len(env_ids), env.device, facts.geometry

    puck_on_their_half = torch.zeros(n, dtype=torch.bool, device=device)
    if "opponent" in env.action_manager.active_terms:
        opponent = env.action_manager.get_term("opponent")
        choice = torch.randint(len(opponent.skill_names), (n,), device=device)
        opponent.choice[env_ids] = choice
        shooter = torch.tensor([name in SHOOTERS for name in opponent.skill_names], device=device)
        puck_on_their_half = shooter[choice]

    x = torch.empty(n, device=device).uniform_(*START_X_FRACTION) * g.rail_x
    x = torch.where(puck_on_their_half, x, -x)  # our half is -x
    y = torch.empty(n, device=device).uniform_(-START_Y_FRACTION, START_Y_FRACTION) * g.puck_limit_y
    entity = env.scene[ENTITY]
    joint_ids, _ = entity.find_joints(("puck_0_x", "puck_0_y"), preserve_order=True)
    entity.write_joint_state_to_sim(
        torch.stack([x, y], dim=-1), torch.zeros(n, 2, device=device), joint_ids, env_ids
    )
