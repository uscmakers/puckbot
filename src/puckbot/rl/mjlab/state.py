"""Table state as batched tensors: where the pucks and paddles are, read from mjlab's entity data.

Usage:
    state = TableState(env, facts).read()
    state.puck_pos      # (num_envs, pucks, 2)      state.paddle_pos   # (num_envs, paddles, 2)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from puckbot.rl.mjlab.scene import ENTITY, TableFacts
from puckbot.rl.sim.skills import Snapshot

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class TableState:
    def __init__(self, env: ManagerBasedRlEnv, facts: TableFacts):
        self._entity = env.scene[ENTITY]
        self._facts = facts
        puck_names = [f"puck_{p}_{axis}" for p in range(facts.n_pucks) for axis in "xy"]
        paddle_names = [f"paddle_{k}_{axis}" for k in range(facts.n_paddles) for axis in "xy"]
        self._puck_ids, _ = self._entity.find_joints(puck_names, preserve_order=True)
        self._paddle_ids, _ = self._entity.find_joints(paddle_names, preserve_order=True)

    def read(self) -> Snapshot:
        data = self._entity.data
        n = data.joint_pos.shape[0]
        return Snapshot(
            puck_pos=data.joint_pos[:, self._puck_ids].reshape(n, -1, 2),
            puck_vel=data.joint_vel[:, self._puck_ids].reshape(n, -1, 2),
            paddle_pos=data.joint_pos[:, self._paddle_ids].reshape(n, -1, 2),
            paddle_vel=data.joint_vel[:, self._paddle_ids].reshape(n, -1, 2),
        )

    @property
    def puck_joint_ids(self) -> list[int]:
        return self._puck_ids

    @property
    def paddle_joint_ids(self) -> list[int]:
        return self._paddle_ids


def goal_flags(puck_pos: torch.Tensor, facts: TableFacts) -> tuple[torch.Tensor, torch.Tensor]:
    """(scored, conceded), each (num_envs,): any puck past the +x / -x slot exit. Robot defends -x."""
    g = facts.geometry
    x, y = puck_pos[..., 0], puck_pos[..., 1]
    in_slot = y.abs() < g.goal_half_width
    scored = (in_slot & (x > g.goal_exit_x)).any(dim=-1)
    conceded = (in_slot & (x < -g.goal_exit_x)).any(dim=-1)
    return scored, conceded
