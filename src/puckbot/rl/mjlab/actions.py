"""Action terms: how the policy, or a scripted opponent, moves a paddle.

PaddleAction (the policy, paddle 0):
    (dx, dy) in [-1, 1] nudges the paddle's goal from where its commanded point is now:
    +-1 = one control tick at top speed (3 m/s x 20 ms = 6 cm), 0 = hold. Same as the CPU sim.
OpponentAction (the opponent, paddle 1):
    a scripted skill (sim/skills.py, the same code the CPU sim uses) picks the goal; each world plays
    one skill per episode, chosen at reset (events.py). The policy does not control it.

Both then run the planner (planner.py) every physics tick and hand its point to the paddle's spring.

Usage (in the env config):
    actions={
        "paddle": PaddleActionCfg(facts=facts),
        "opponent": OpponentActionCfg(facts=facts, skills=("goalie", "straight_shot", "bank_shot")),
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from puckbot.rl.mjlab.planner import BatchedPlanner
from puckbot.rl.mjlab.scene import ENTITY, TableFacts
from puckbot.rl.mjlab.state import TableState
from puckbot.rl.sim.skills import Skill, make_skill

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ── Shared: a goal, the planner, the spring ───────────────────────────────────


class _PlannedPaddle(ActionTerm):
    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        facts, k = cfg.facts, cfg.paddle
        self._joint_ids, _ = self._entity.find_joints((f"paddle_{k}_x", f"paddle_{k}_y"), preserve_order=True)
        paddle = facts.settings.paddle
        self._planner = BatchedPlanner(
            self.num_envs,
            facts.paddle_low[k],
            facts.paddle_high[k],
            paddle.max_speed_mps,
            paddle.max_accel_mps2,
            self.device,
        )
        self._home = torch.as_tensor(facts.home[k], dtype=torch.float32, device=self.device)
        self._goal = self._home.repeat(self.num_envs, 1)
        self._physics_dt = env.physics_dt
        self.reset()

    @property
    def command_point(self) -> torch.Tensor:
        return self._planner.position

    def apply_actions(self) -> None:  # every physics tick
        self._planner.advance(self._goal, self._physics_dt)
        self._entity.set_joint_position_target(self._planner.position, joint_ids=self._joint_ids)

    def reset(self, env_ids=None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._planner.reset(env_ids, self._home)  # every episode starts with the paddles at home
        self._goal[env_ids] = self._home


# ── The policy's paddle ───────────────────────────────────────────────────────


@dataclass(kw_only=True)
class PaddleActionCfg(ActionTermCfg):
    facts: TableFacts
    paddle: int = 0
    entity_name: str = ENTITY

    def build(self, env: ManagerBasedRlEnv) -> PaddleAction:
        return PaddleAction(self, env)


class PaddleAction(_PlannedPaddle):
    def __init__(self, cfg: PaddleActionCfg, env: ManagerBasedRlEnv):
        self._raw = torch.zeros(env.num_envs, 2, device=env.device)
        super().__init__(cfg, env)
        self._full_nudge_m = cfg.facts.settings.paddle.max_speed_mps * env.step_dt

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:  # once per control tick
        self._raw[:] = actions
        self._goal = self._planner.position + actions.clamp(-1.0, 1.0) * self._full_nudge_m

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._raw[slice(None) if env_ids is None else env_ids] = 0.0


# ── The opponent's paddle: a scripted skill, one chosen per world ─────────────


@dataclass(kw_only=True)
class OpponentActionCfg(ActionTermCfg):
    facts: TableFacts
    skills: tuple[str, ...]  # the skills it can play (sim/skills.py); events pick one per world each episode
    paddle: int = 1
    entity_name: str = ENTITY

    def build(self, env: ManagerBasedRlEnv) -> OpponentAction:
        return OpponentAction(self, env)


class OpponentAction(_PlannedPaddle):
    def __init__(self, cfg: OpponentActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.skill_names = cfg.skills
        self._skills: list[Skill] = [
            make_skill(name, cfg.facts, cfg.paddle, self.device) for name in cfg.skills
        ]
        self.choice = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)  # index into skills
        self._state = TableState(env, cfg.facts)
        self._empty = torch.zeros(env.num_envs, 0, device=env.device)
        self._rows = torch.arange(env.num_envs, device=env.device)

    @property
    def action_dim(self) -> int:
        return 0  # the policy does not control it

    @property
    def raw_action(self) -> torch.Tensor:
        return self._empty

    def process_actions(self, actions: torch.Tensor) -> None:  # once per control tick
        snapshot = self._state.read()
        every = torch.stack([skill.target(snapshot) for skill in self._skills])  # (skills, worlds, 2)
        self._goal = every[self.choice, self._rows]

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        for skill in getattr(self, "_skills", ()):  # the base class resets once before the skills exist
            skill.reset(env_ids)
