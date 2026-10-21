"""The game task in mjlab: our policy plays against a scripted opponent.

    opponent   each world draws one skill per episode from `game.opponents` (train.json); with one
               paddle there is no opponent
    start      a shooter opponent gets the puck on its half (we defend); the goalie, or no opponent,
               leaves the puck on our half (we attack); our paddle starts at home (events.py)
    reward     +1 scored, -1 conceded, plus potential-based shaping for closing the distance between
               our paddle and the puck (a gap closed and reopened nets zero, so it cannot be farmed)
    ends       a goal either way, or `episode_seconds`

Physics: 0.5 ms ticks (rail bounces need it), 40 per control tick = 50 Hz, Euler integrator like the
CPU sim. PPO: mjlab / RSL-RL defaults (only the run length, name and logger are ours).

Usage:
    env_cfg = game_env_cfg(num_envs=4096)
    runner_cfg = game_ppo_cfg()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import last_action, reset_scene_to_default, time_out
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from puckbot.rl.mjlab.actions import OpponentActionCfg, PaddleActionCfg
from puckbot.rl.mjlab.events import start_episode
from puckbot.rl.mjlab.observations import OwnPaddle, camera_frames
from puckbot.rl.mjlab.scene import TableFacts, make_scene_cfg
from puckbot.rl.mjlab.state import TableState, goal_flags
from puckbot.rl.sim.layout import Layout
from puckbot.rl.train.settings import GameSettings, load_train_settings

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

DISCOUNT = 0.99  # PPO's discount; the shaping is potential-based with it


# ── Judge: one place that works out what happened this tick ───────────────────


class GameJudge:
    """Scored / conceded and the shaping term, once per tick; rewards and terminations both read it."""

    def __init__(self, env: ManagerBasedRlEnv, facts: TableFacts):
        self.facts = facts
        self.state = TableState(env, facts)
        self.distance = torch.zeros(env.num_envs, device=env.device)
        self.last_step = -1

    def update(self, env: ManagerBasedRlEnv) -> GameJudge:
        if self.last_step != env.common_step_counter:
            self.last_step = env.common_step_counter
            snap = self.state.read()
            self.scored, self.conceded = goal_flags(snap.puck_pos, self.facts)
            distance = (snap.paddle_pos[:, 0] - snap.puck_pos[:, 0]).norm(dim=-1)
            first_tick = env.episode_length_buf <= 1
            self.shaping = torch.where(
                first_tick, torch.zeros_like(distance), self.distance - DISCOUNT * distance
            )
            self.distance = distance
        return self


def _judge(env: ManagerBasedRlEnv, facts: TableFacts) -> GameJudge:
    if not hasattr(env, "_game_judge"):
        env._game_judge = GameJudge(env, facts)
    return env._game_judge.update(env)


def scored(env, facts: TableFacts) -> torch.Tensor:
    return _judge(env, facts).scored


def conceded(env, facts: TableFacts) -> torch.Tensor:
    return _judge(env, facts).conceded


def approach(env, facts: TableFacts) -> torch.Tensor:
    return _judge(env, facts).shaping


# ── Env and PPO config ────────────────────────────────────────────────────────


def game_env_cfg(num_envs: int = 4096, play: bool = False) -> ManagerBasedRlEnvCfg:
    settings = load_train_settings()
    game: GameSettings = settings.game
    if (game.paddles > 1) != bool(game.opponents):
        raise ValueError("name opponents exactly when there is more than one paddle")
    facts = TableFacts.load(Layout.of(game.paddles, game.pucks))
    f = {"facts": facts}

    actions = {"paddle": PaddleActionCfg(facts=facts)}
    if game.paddles > 1:
        actions["opponent"] = OpponentActionCfg(facts=facts, skills=tuple(game.opponents))

    policy_terms = {
        "camera": ObservationTermCfg(
            func=camera_frames, params={**f, "noise_m": settings.camera.puck_noise_m}
        ),
        "own_paddle": ObservationTermCfg(func=OwnPaddle, params=f),
        "last_action": ObservationTermCfg(func=last_action, params={"action_name": "paddle"}),
    }
    return ManagerBasedRlEnvCfg(
        scene=make_scene_cfg(facts, num_envs, settings.camera),
        sim=SimulationCfg(
            nconmax=32, njmax=128, mujoco=MujocoCfg(timestep=facts.physics_dt, integrator="euler")
        ),
        decimation=facts.decimation,
        episode_length_s=game.episode_seconds if not play else 1e9,
        scale_rewards_by_dt=False,  # our rewards are per event, not per second
        observations={
            "actor": ObservationGroupCfg(policy_terms, enable_corruption=not play),
            "critic": ObservationGroupCfg(dict(policy_terms)),
        },
        actions=actions,
        events={  # order matters: everything back to its start (paddles home), then the episode's start
            "reset_scene": EventTermCfg(func=reset_scene_to_default, mode="reset"),
            "start_episode": EventTermCfg(func=start_episode, mode="reset", params=f),
        },
        rewards={
            "scored": RewardTermCfg(func=scored, weight=game.scored_reward, params=f),
            "conceded": RewardTermCfg(func=conceded, weight=game.conceded_reward, params=f),
            "approach": RewardTermCfg(func=approach, weight=game.approach_reward_per_m, params=f),
        },
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            "scored": TerminationTermCfg(func=scored, params=f),
            "conceded": TerminationTermCfg(func=conceded, params=f),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.0, 0.0, 0.74),
            distance=2.2,
            elevation=-90.0,
            azimuth=90.0,
            width=640,
            height=360,
        ),
    )


def game_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """mjlab / RSL-RL PPO defaults; only the run length, name and logger are ours."""
    return RslRlOnPolicyRunnerCfg(experiment_name="puckbot_game", max_iterations=1500, logger="tensorboard")
