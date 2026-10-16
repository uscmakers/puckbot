"""Training setup: camera, action, serves, the game env (any layout, any task), and a few PPO steps."""

from dataclasses import replace

import numpy as np
import pytest

from puckbot.rl.sim.env import AirHockeyEnv
from puckbot.rl.train.action import action_to_target, target_to_action
from puckbot.rl.train.camera import CameraModel
from puckbot.rl.train.evaluate import evaluate, skill_policy
from puckbot.rl.train.game_env import GameEnv
from puckbot.rl.train.serves import random_serve
from puckbot.rl.train.settings import CameraSettings, load_train_settings
from puckbot.rl.train.tasks import Verdict


@pytest.fixture(scope="module")
def settings():
    return load_train_settings()


def with_game(settings, **game):
    return replace(settings, game=replace(settings.game, **game))


# ── Camera ────────────────────────────────────────────────────────────────────


def _camera(latency_s=0.02, lags=(0, 3), noise=0.0, objects=1):
    settings = CameraSettings(rate_hz=100, latency_s=latency_s, puck_noise_m=noise, history_lags_frames=lags)
    return CameraModel(settings, objects, physics_dt=0.001, random=np.random.default_rng(0))


def _record_straight_line(camera, seconds, objects=1):
    """Every object moves along x at 1 m/s: its x equals the time."""
    times = np.arange(1, round(seconds / 0.001) + 1) * 0.001
    one = np.stack([times, np.zeros_like(times)], axis=1)
    camera.record(np.stack([one] * objects, axis=1))


def test_camera_frames_are_delayed_and_in_order():
    camera = _camera()
    camera.reset([(0.0, 0.0)])
    _record_straight_line(camera, 0.2)
    newest, older = camera.latest_frames()
    assert newest[0, 0] == pytest.approx(0.18)  # the picture taken at 0.18 s has just arrived
    assert older[0, 0] == pytest.approx(0.15)  # three frames (30 ms) older


def test_camera_before_any_frame_shows_the_start():
    camera = _camera()
    camera.reset([(0.5, 0.1)])
    _record_straight_line(camera, 0.005)
    assert np.allclose(camera.latest_frames(), [[[0.5, 0.1]], [[0.5, 0.1]]])


def test_camera_noise_is_applied():
    camera = _camera(noise=0.01)
    camera.reset([(0.0, 0.0)])
    assert np.abs(camera.latest_frames()).max() > 0


def test_camera_tracks_several_objects():
    camera = _camera(objects=3)
    camera.reset(np.zeros((3, 2)))
    _record_straight_line(camera, 0.2, objects=3)
    assert camera.latest_frames().shape == (2, 3, 2)


# ── Action ────────────────────────────────────────────────────────────────────


def test_zero_action_holds_and_full_action_is_one_tick_of_top_speed():
    env = AirHockeyEnv()
    env.reset()
    assert np.allclose(action_to_target([0, 0], env), env.paddle_target[0])
    nudge = action_to_target([1, 0], env) - env.paddle_target[0]
    assert nudge[0] == pytest.approx(env.settings.paddle.max_speed_mps * env.step_seconds)


def test_action_and_target_convert_back_and_forth():
    env = AirHockeyEnv()
    env.reset()
    assert np.allclose(target_to_action(action_to_target([0.3, -0.5], env), env), [0.3, -0.5])


# ── Serves ────────────────────────────────────────────────────────────────────


def test_every_serve_heads_for_our_end_of_the_table(settings):
    geometry = AirHockeyEnv().geometry
    random = np.random.default_rng(0)
    serves = [random_serve(random, geometry, settings.goalie) for _ in range(200)]
    assert all(serve.puck_vel[0] < 0 and serve.puck_pos[0] > 0 for serve in serves)
    low, high = settings.goalie.puck_speed_mps
    assert all(low - 1e-9 <= np.linalg.norm(serve.puck_vel) <= high + 1e-9 for serve in serves)
    assert any(serve.banks for serve in serves) and not all(serve.banks for serve in serves)


# ── Game env ──────────────────────────────────────────────────────────────────


def test_env_reset_and_step_shapes(settings):
    env = GameEnv(settings)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (16,)
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape and np.isfinite(reward)
    assert env.observation_space.contains(obs)


def test_same_seed_gives_the_same_serve_and_observations(settings):
    env = GameEnv(settings)
    first, _ = env.reset(seed=3)
    second, _ = env.reset(seed=3)
    assert np.allclose(first, second)
    after_first = env.step([0.2, 0.1])[0]
    env.reset(seed=3)
    assert np.allclose(env.step([0.2, 0.1])[0], after_first)


def test_scripted_goalie_beats_doing_nothing(settings):
    outcomes = {
        name: evaluate(skill_policy(name), episodes=30, env=GameEnv(settings)) for name in ("hold", "goalie")
    }
    assert outcomes["goalie"].counts.get("conceded", 0) < outcomes["hold"].counts.get("conceded", 0)
    assert outcomes["goalie"].block_rate > 0.9


def test_every_episode_ends_with_an_outcome(settings):
    env = GameEnv(settings)
    env.reset(seed=1)
    for _ in range(env.task.max_ticks + 1):
        _, _, terminated, truncated, info = env.step([0, 0])
        if terminated or truncated:
            assert info["outcome"] in ("blocked", "conceded", "missed", "timeout")
            return
    pytest.fail("episode never ended")


# ── One env, any layout and task ──────────────────────────────────────────────


def test_a_second_paddle_adds_the_opponent_to_what_the_camera_sees(settings):
    with_opponent = with_game(settings, paddles=2, opponents=("goalie", "hold"))
    env = GameEnv(with_opponent)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (26,)  # 5 lags x (puck + opponent paddle) x 2 + our paddle 4 + last action 2
    for _ in range(5):
        env.step(env.action_space.sample())
    assert env.sim.paddle_target.shape == (2, 2)


def test_opponents_are_named_exactly_when_there_is_more_than_one_paddle(settings):
    with pytest.raises(ValueError, match="opponents"):
        GameEnv(with_game(settings, paddles=2))
    with pytest.raises(ValueError, match="opponents"):
        GameEnv(with_game(settings, opponents=("goalie",)))


def test_opponents_are_drawn_per_episode_from_the_named_skills(settings):
    env = GameEnv(with_game(settings, paddles=2, opponents=("goalie", "hold")))
    names = set()
    for seed in range(20):
        env.reset(seed=seed)
        names.add(type(env.active[1]).__name__)
    assert names == {"Goalie", "Hold"}


def test_the_env_runs_any_task(settings):
    class FiveTicksThenDone:
        max_ticks = 5

        def reset(self, random, sim):
            return sim.reset(puck_pos=(0.3, 0.0)), {}

        def judge(self, state, tick):
            return Verdict(reward=1.0, outcome="done" if tick == 5 else None)

    env = GameEnv(settings, task=FiveTicksThenDone())
    env.reset(seed=0)
    steps = 0
    done = False
    while not done:
        _, reward, done, _, info = env.step([0, 0])
        steps += 1
        assert reward == 1.0
    assert steps == 5 and info["outcome"] == "done"


# ── PPO ───────────────────────────────────────────────────────────────────────


def test_ppo_runs_a_few_steps(settings, tmp_path):
    from puckbot.rl.train.train import train

    tiny = replace(settings, training=replace(settings.training, parallel_envs=1, total_steps=256))
    model = train(tiny, tmp_path)
    assert (tmp_path / "goalie_ppo.zip").exists()
    assert model.policy is not None
