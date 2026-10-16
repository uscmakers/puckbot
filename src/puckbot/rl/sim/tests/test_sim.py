"""The sim: model, paddle motion, physics sanity, and the hand-written skills."""

import numpy as np
import pytest

from puckbot.rl.sim import baselines
from puckbot.rl.sim.baselines import shot_grid
from puckbot.rl.sim.drills import DRILLS, run_drill, run_skill
from puckbot.rl.sim.env import AirHockeyEnv
from puckbot.rl.sim.layout import Box, Layout, PaddleSetup
from puckbot.rl.sim.model import paddle_limits
from puckbot.rl.sim.settings import load_settings
from puckbot.rl.sim.skills import make_skill


@pytest.fixture(scope="module")
def env():
    return AirHockeyEnv()


def test_table_size_matches_cad(env):
    g = env.geometry
    assert g.rail_x == pytest.approx(1.0448)  # playing surface 2.0896 x 1.078 m
    assert g.rail_y == pytest.approx(0.539)
    assert g.goal_half_width == pytest.approx(0.1238)


def test_paddle_stays_on_our_side(env):
    assert env.paddle_high[0, 0] < 0
    assert env.home[0, 0] > env.paddle_low[0, 0]


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


def test_paddle_arrives_without_overshoot(env):
    env.reset()
    worst_overshoot = 0.0
    for _ in range(150):
        state = env.step((-0.3, 0.4))
        worst_overshoot = max(worst_overshoot, state.paddle_pos[0, 0] - -0.3, state.paddle_pos[0, 1] - 0.4)
    assert worst_overshoot < 0.003


def test_paddle_turns_around_without_ringing(env):
    env.reset()
    for _ in range(60):
        env.step((-0.3, 0.4))
    for _ in range(100):
        state = env.step((-0.3, -0.4))
    assert np.allclose(state.paddle_pos, (-0.3, -0.4), atol=0.003)


def test_the_paddle_is_a_mass_on_a_critically_damped_mujoco_spring(env):
    paddle_mass = float(env.model.body_mass[env.model.body("paddle_0").id])
    omega = 1 / env.settings.paddle.response_time_s
    assert env.model.nu == 2 * env.n_paddles
    assert env.model.actuator_gainprm[0, 0] == pytest.approx(paddle_mass * omega**2)
    assert env.model.actuator_biasprm[0, 2] == pytest.approx(-2 * paddle_mass * omega)


def test_target_point_obeys_the_acceleration_and_speed_limits(env):
    caps = env.settings.paddle
    env.reset()
    env.step((-0.1, 0.4))  # a far diagonal target
    first_step = env.paddle_target[0] - env.home[0]
    assert np.linalg.norm(first_step) == pytest.approx(
        0.5 * caps.max_accel_mps2 * env.step_seconds**2, rel=0.05
    )
    assert first_step[1] / first_step[0] == pytest.approx((0.4 - 0.0) / (-0.1 - env.home[0, 0]), rel=0.01)
    speeds = []
    for _ in range(40):
        before = env.paddle_target[0].copy()
        env.step((-0.1, 0.4))
        speeds.append(np.linalg.norm(env.paddle_target[0] - before) / env.step_seconds)
    assert max(speeds) == pytest.approx(caps.max_speed_mps, rel=0.02)


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
        if state.puck_vel[0, 1] < 0:
            break
    assert 0.2 < -state.puck_vel[0, 1] / 3.0 < 0.95


def test_puck_never_escapes_or_speeds_up_off_the_rails(env):
    """Random shots away from the goals: every one stays on the table and none gains speed."""
    rng = np.random.default_rng(0)
    for _ in range(40):
        angle, speed = rng.uniform(0, 2 * np.pi), rng.uniform(1, 8)
        start = (rng.uniform(-0.6, 0.6), rng.uniform(-0.4, 0.4))
        vel = (speed * np.cos(angle), speed * np.sin(angle))
        state = env.reset(puck_pos=start, puck_vel=vel, paddle_pos=(-0.9, 0.4))
        for _ in range(100):
            state = env.step(env.home)
            if state.goal or abs(state.puck_pos[0, 0]) > 0.9:  # goal ends are checked by the drills
                break
            assert abs(state.puck_pos[0, 1]) < env.geometry.rail_y
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
    assert np.any(np.diff(np.sign(run.puck_vel[:, 0, 1])) != 0)  # sideways direction flipped = rail hit


def test_goalie_blocks_the_shot(env):
    run = run_drill(env, "goalie")
    touch = env.geometry.puck_radius + env.geometry.paddle_radius
    assert run.outcome != "conceded"
    assert (
        np.linalg.norm(run.puck_pos - run.paddle_pos, axis=-1).min() < touch + 0.005
    )  # it got a paddle on it


def test_every_drill_has_a_skill():
    assert set(DRILLS) == {"hold", "straight_shot", "bank_shot", "goalie"}


# ── Baselines ─────────────────────────────────────────────────────────────────


def test_run_skill_from_any_start_matches_the_drill(env):
    drill = DRILLS["straight_shot"]
    by_hand = run_skill(env, "straight_shot", drill.puck_pos, drill.puck_vel, drill.seconds)
    assert by_hand.outcome == run_drill(env, "straight_shot").outcome == "scored"


def test_shot_grid_reports_every_start(env, monkeypatch):
    monkeypatch.setattr(baselines, "START_X_M", (-0.40,))
    monkeypatch.setattr(baselines, "START_Y_M", (-0.2, 0.2))
    result = shot_grid(env, "straight_shot")
    assert result.outcomes.shape == (2, 1) and result.scored.all()
    assert "scored 2/2" in str(result)


# ── Layout: any number of paddles and pucks ───────────────────────────────────


def test_default_layout_is_one_robot_paddle_and_one_puck():
    layout = Layout()
    assert layout.pucks == 1 and [setup.side for setup in layout.paddles] == ["robot"]
    assert [setup.side for setup in Layout.of(2).paddles] == ["robot", "opponent"]
    with pytest.raises(ValueError):
        Layout.of(0)


@pytest.mark.parametrize("paddles, pucks", [(1, 1), (2, 1), (2, 3), (1, 2)])
def test_state_shapes_follow_the_layout(paddles, pucks):
    env = AirHockeyEnv(layout=Layout.of(paddles, pucks))
    env.reset()
    state = env.step(np.zeros((paddles, 2)))
    assert state.puck_pos.shape == state.puck_vel.shape == (pucks, 2)
    assert state.paddle_pos.shape == state.paddle_vel.shape == (paddles, 2)
    assert len(state.goals) == pucks and state.touches.shape == (paddles, pucks)
    assert env.puck_trace.shape == (env.physics_ticks_per_step, pucks, 2)


def test_the_opponent_paddle_is_our_mirror_image():
    env = AirHockeyEnv(layout=Layout.of(2))
    assert np.allclose(env.paddle_low[1], [-env.paddle_high[0, 0], env.paddle_low[0, 1]])
    assert np.allclose(env.paddle_high[1], [-env.paddle_low[0, 0], env.paddle_high[0, 1]])
    assert np.allclose(env.home[1], env.home[0] * [-1, 1])


def test_reach_can_be_the_full_table_or_a_box(env):
    settings, geometry = env.settings, env.geometry
    full = paddle_limits(PaddleSetup(reach="full_table"), settings, geometry)
    assert full[0][0] < 0 < full[1][0] and full[0][0] == -full[1][0]

    box = AirHockeyEnv(layout=Layout((PaddleSetup(reach=Box(-0.6, -0.2, -0.3, 0.2)),)))
    assert np.allclose(box.paddle_low[0], [-0.6, -0.2]) and np.allclose(box.paddle_high[0], [-0.3, 0.2])
    assert np.all(box.home >= box.paddle_low) and np.all(box.home <= box.paddle_high)  # pulled inside

    with pytest.raises(ValueError, match="full reach"):
        paddle_limits(PaddleSetup(reach=Box(-2.0, -0.2, -0.3, 0.2)), settings, geometry)


def test_wrong_number_of_rows_is_a_clear_error(env):
    with pytest.raises(ValueError, match="puck_pos: expected 1 row"):
        env.reset(puck_pos=[(0, 0), (1, 1)])
    with pytest.raises(ValueError, match="targets: expected 1 row"):
        env.step([(0, 0), (1, 1)])


def test_extra_paddles_and_pucks_do_not_change_the_physics(env):
    """The same shot with an idle opponent paddle, or a second puck far away, plays out the same."""
    alone = run_skill(env, "bank_shot", (-0.40, 0.15), (0, 0), 3.0)

    with_opponent = run_skill(AirHockeyEnv(layout=Layout.of(2)), "bank_shot", (-0.40, 0.15), (0, 0), 3.0)
    assert np.allclose(with_opponent.puck_pos[:, 0], alone.puck_pos[:, 0], atol=1e-9)
    assert np.allclose(with_opponent.paddle_pos[:, 0], alone.paddle_pos[:, 0], atol=1e-9)

    two_pucks = AirHockeyEnv(layout=Layout.of(1, pucks=2))
    state = two_pucks.reset(puck_pos=[(-0.40, 0.15), (0.2, -0.45)])
    skill = make_skill("bank_shot", two_pucks)
    for _ in range(len(alone.time) - 1):
        state = two_pucks.step(skill.target(state))
    # Not bit-for-bit: MuJoCo solves all contacts together and stops at a tolerance, so a resting puck's
    # contacts nudge the result by ~1e-6, which a bank shot's bounce grows to a few millimetres.
    assert np.allclose(state.puck_pos[0], alone.puck_pos[-1, 0], atol=0.02)
    assert state.goal == alone.outcome[:6]


def test_touches_say_which_paddle_hit_which_puck():
    env = AirHockeyEnv(layout=Layout.of(2, pucks=2))
    state = env.reset(puck_pos=[(-0.5, 0.0), (0.5, 0.3)], puck_vel=[(-3.0, 0.0), (0.0, 0.0)])
    touched = np.zeros((2, 2), dtype=bool)
    for _ in range(40):
        state = env.step(env.home)
        touched |= state.touches
    assert touched[0, 0] and not touched[1].any() and not touched[0, 1]


def test_skills_play_either_end():
    """The goalie of the opponent end blocks a shot from the robot: the skills mirror for that paddle."""
    env = AirHockeyEnv(layout=Layout.of(2))
    run = run_skill(env, "straight_shot", (-0.40, 0.15), (0, 0), 4.0, opponent="goalie")
    assert run.outcome != "scored"
    assert run.paddle_pos[:, 1, 0].min() > 0  # it stayed on its own side
    assert abs(run.paddle_pos[:, 1, 1]).max() > 0.01  # and moved to meet the puck
