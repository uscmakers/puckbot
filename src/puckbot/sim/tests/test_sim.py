"""The sim: model, paddle motion, physics sanity, and the hand-written skills."""

import numpy as np
import pytest

from puckbot.sim.drills import DRILLS, run_drill
from puckbot.sim.env import AirHockeyEnv
from puckbot.sim.settings import load_settings


@pytest.fixture(scope="module")
def env():
    return AirHockeyEnv()


# ── Model ─────────────────────────────────────────────────────────────────────


def test_table_size_matches_cad(env):
    g = env.geometry
    assert g.rail_x == pytest.approx(1.0448)  # playing surface 2.0896 x 1.078 m
    assert g.rail_y == pytest.approx(0.539)
    assert g.goal_half_width == pytest.approx(0.1238)


def test_paddle_stays_on_our_side(env):
    assert env.paddle_high[0] < 0
    assert env.home[0] > env.paddle_low[0]


def test_settings_typo_is_an_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"control_rate_hz": 50, "oops": 1}')
    with pytest.raises(ValueError, match="oops"):
        load_settings(bad)


# ── Paddle ────────────────────────────────────────────────────────────────────


def test_paddle_reaches_target_within_speed_limit(env):
    caps = env.settings.paddle
    state = env.reset()
    top_speed = 0.0
    for _ in range(100):
        state = env.step((-0.3, 0.4))
        top_speed = max(top_speed, np.linalg.norm(state.paddle_vel))
    assert np.allclose(state.paddle_pos, (-0.3, 0.4), atol=0.005)
    assert top_speed < caps.max_speed_mps * 1.1


def test_paddle_target_outside_limits_is_clamped(env):
    env.reset()
    for _ in range(100):
        state = env.step((5.0, 5.0))
    assert np.all(state.paddle_pos <= env.paddle_high + 1e-3)


# ── Puck physics ──────────────────────────────────────────────────────────────


def test_puck_rests_when_not_pushed(env):
    state = env.reset(puck_pos=(0.2, -0.1))
    for _ in range(50):
        state = env.step(env.home)
    assert np.allclose(state.puck_pos, (0.2, -0.1), atol=1e-6)


def test_puck_down_the_middle_is_a_goal(env):
    state = env.reset(puck_pos=(-0.2, 0.0), puck_vel=(3.0, 0.0))
    for _ in range(100):
        state = env.step(env.home)
        if state.goal:
            break
    assert state.goal == "scored"


def test_puck_bounces_off_side_rail_and_loses_some_speed(env):
    state = env.reset(puck_pos=(0.3, 0.0), puck_vel=(0.0, 3.0))
    for _ in range(100):
        state = env.step(env.home)
        if state.puck_vel[1] < 0:
            break
    assert 0.2 < -state.puck_vel[1] / 3.0 < 0.95


def test_puck_never_escapes_or_speeds_up_off_the_rails(env):
    """Random shots away from the goals: every one stays on the table and none gains speed."""
    rng = np.random.default_rng(0)
    for _ in range(40):
        angle, speed = rng.uniform(0, 2 * np.pi), rng.uniform(1, 8)
        start = (rng.uniform(-0.6, 0.6), rng.uniform(-0.4, 0.4))
        vel = (speed * np.cos(angle), speed * np.sin(angle))
        state = env.reset(puck_pos=start, puck_vel=vel, paddle_pos=(-0.9, 0.4))
        for _ in range(100):
            state = env.step((-0.9, 0.4))
            if state.goal or abs(state.puck_pos[0]) > 0.9:  # goal ends are checked by the drills
                break
            assert abs(state.puck_pos[1]) < env.geometry.rail_y
            assert np.linalg.norm(state.puck_vel) < speed * 1.01


# ── Skills ────────────────────────────────────────────────────────────────────


def test_hold_leaves_the_puck_alone(env):
    run = run_drill(env, "hold")
    assert run.outcome == "no goal"
    assert run.top_puck_speed == 0.0
    assert np.allclose(run.paddle_pos[-1], env.home, atol=0.005)


def test_straight_shot_scores(env):
    assert run_drill(env, "straight_shot").outcome == "scored"


def test_bank_shot_scores_after_a_rail_bounce(env):
    run = run_drill(env, "bank_shot")
    assert run.outcome == "scored"
    assert np.any(np.diff(np.sign(run.puck_vel[:, 1])) != 0)  # sideways direction flipped = rail hit


def test_goalie_blocks_the_shot(env):
    run = run_drill(env, "goalie")
    touch = env.geometry.puck_radius + env.geometry.paddle_radius
    assert run.outcome != "conceded"
    assert (
        np.linalg.norm(run.puck_pos - run.paddle_pos, axis=1).min() < touch + 0.005
    )  # it got a paddle on it


def test_every_drill_has_a_skill():
    assert set(DRILLS) == {"hold", "straight_shot", "bank_shot", "goalie"}
