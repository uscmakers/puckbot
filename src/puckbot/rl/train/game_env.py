"""The game as a Gymnasium environment. One env for every task.

The agent drives paddle 0 with (dx, dy). Any other paddles are played by scripted opponents. The
number of paddles and pucks, and which opponents, come from the settings (`game`); the task (tasks.py)
decides where things start, the reward, and when an episode ends.

Steps (every control tick, 50 Hz):
    1. Act      the policy's (dx, dy) nudges paddle 0's commanded point (action.py);
                each opponent skill picks its own paddle's target
    2. Move     the sim runs 20 ms; the camera model records the pucks and the opponent paddles
    3. Judge    the task returns the reward and whether the episode is over
    4. Observe  newest camera frames + our paddle + previous action (observation.py)

Opponents are scripted and read the true state (no camera delay or noise yet).

Usage:
    env = GameEnv()                                   # goalie task, one paddle, one puck
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    info["outcome"]        # on the last step, whatever the task calls it ("blocked", "timeout", ...)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from puckbot.rl.sim.env import AirHockeyEnv
from puckbot.rl.sim.layout import Layout
from puckbot.rl.sim.settings import Settings, load_settings
from puckbot.rl.sim.skills import make_skill
from puckbot.rl.train.action import action_to_target
from puckbot.rl.train.camera import CameraModel
from puckbot.rl.train.observation import ObservationEncoder
from puckbot.rl.train.settings import TrainSettings, load_train_settings
from puckbot.rl.train.tasks import GoalieTask, Task


class GameEnv(gym.Env):
    def __init__(
        self,
        train_settings: TrainSettings | None = None,
        sim_settings: Settings | None = None,
        task: Task | None = None,
    ):
        self.settings = train_settings or load_train_settings()
        game = self.settings.game
        if (game.paddles > 1) != bool(game.opponents):
            raise ValueError("name opponents exactly when there is more than one paddle")

        self.sim = AirHockeyEnv(sim_settings or load_settings(), Layout.of(game.paddles, game.pucks))
        self.task = task or GoalieTask(self.settings.goalie, self.sim)
        self.opponents = {  # paddle number -> {skill name -> skill}
            paddle: {name: make_skill(name, self.sim, paddle) for name in game.opponents}
            for paddle in range(1, game.paddles)
        }

        tracked = game.pucks + game.paddles - 1  # the camera sees every puck and every opponent paddle
        lags = len(self.settings.camera.history_lags_frames)
        self.camera = CameraModel(self.settings.camera, tracked, self.sim.physics_dt, random=self.np_random)
        self.encoder = ObservationEncoder(self.sim.geometry, self.sim.settings)
        self.observation_space = gym.spaces.Box(
            -2.0, 2.0, (self.encoder.size(lags, tracked),), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

    # ── Gymnasium API ──

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.camera.random = self.np_random
        self.state, info = self.task.reset(self.np_random, self.sim)
        self.camera.reset(self._tracked(self.state.puck_pos, self.state.paddle_pos))
        self.active = {paddle: self._pick(skills) for paddle, skills in self.opponents.items()}
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.ticks = 0
        return self._observation(), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        targets = np.empty((self.sim.n_paddles, 2))
        targets[0] = action_to_target(action, self.sim)
        for paddle, skill in self.active.items():
            targets[paddle] = skill.target(self.state)

        self.state = self.sim.step(targets)
        self.camera.record(self._tracked(self.sim.puck_trace, self.sim.paddle_trace))
        self.ticks += 1
        self.previous_action = action

        verdict = self.task.judge(self.state, self.ticks)
        terminated = verdict.outcome is not None and not verdict.truncated
        info = {"outcome": verdict.outcome} if verdict.outcome else {}
        return self._observation(), verdict.reward, terminated, verdict.truncated, info

    # ── Helpers ──

    def _pick(self, skills: dict):
        name = self.np_random.choice(sorted(skills))
        skill = skills[name]
        skill.reset()
        return skill

    @staticmethod
    def _tracked(pucks: np.ndarray, paddles: np.ndarray) -> np.ndarray:
        """What the camera sees: the pucks, then every paddle but ours. Also works on a trace."""
        return np.concatenate([pucks, paddles[..., 1:, :]], axis=-2)

    def _observation(self) -> np.ndarray:
        return self.encoder.encode(self.camera.latest_frames(), self.state, self.previous_action)
