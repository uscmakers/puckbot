"""Showcase: two scripted skills play a short match, saved as an mp4 (for showing people the sim).

Steps:
    1. Serve    puck at rest on a random spot of a shooter's half; both paddles at home. Shooters take
                turns serving; a goalie or hold never serves
    2. Play     every control tick each skill picks its paddle's target -> env steps (normal CPU MuJoCo)
    3. Point    ends on a goal, or POINT_SECONDS after the serve (blocked / missed) -> serve again.
                A demo-only referee: the shooters take one shot per serve, and training never restarts
    4. Render   one continuous top-down mp4, and print each point as it happens

Left paddle = robot side (-x), right paddle = opponent side; skills: hold, goalie, straight_shot, bank_shot.

Usage (rendering needs OpenGL: MUJOCO_GL=egl on a GPU node; on a CPU node see rl/mjlab/record.py):
    python -m puckbot.rl.sim.showcase straight_shot goalie          # attacker vs goalie, 20 s
    python -m puckbot.rl.sim.showcase bank_shot straight_shot --seconds 30 --out outputs/showcase.mp4
"""

from __future__ import annotations

import os

if os.environ.get("MUJOCO_GL") == "osmesa":  # software OpenGL must load before anything else
    import OpenGL.GL  # noqa: F401

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

from puckbot.rl.sim.env import AirHockeyEnv  # noqa: E402
from puckbot.rl.sim.layout import Layout  # noqa: E402
from puckbot.rl.sim.skills import SKILL_NAMES, make_skill  # noqa: E402

SERVE_X_FRACTION = (0.3, 0.7)  # serve this far from the centre line (fraction of half the length)
SERVE_Y_FRACTION = 0.6  # and within this fraction of the side-rail limit
POINT_SECONDS = 3.0  # a point with no goal ends this long after the serve
SHOOTERS = ("straight_shot", "bank_shot")


# ── Match ─────────────────────────────────────────────────────────────────────


def play_match(env: AirHockeyEnv, left: str, right: str, seconds: float, seed: int = 0):
    """Returns (qpos per control tick, list of (time, event) lines)."""
    rng = np.random.default_rng(seed)
    skills = [make_skill(left, env, paddle=0), make_skill(right, env, paddle=1)]
    servers = [side for side, name in ((-1, left), (+1, right)) if name in SHOOTERS]
    if not servers:
        raise ValueError(f"no shooter to serve: pick one of {SHOOTERS}")
    g = env.geometry
    score = {left: 0, right: 0} if left != right else {"left": 0, "right": 0}
    names = {-1: left, +1: right} if left != right else {-1: "left", +1: "right"}
    events: list[tuple[float, str]] = []
    qpos: list[np.ndarray] = []

    def serve(point: int):  # the puck goes on the server's half: -1 = left, +1 = right
        side = servers[point % len(servers)]
        x = side * rng.uniform(*SERVE_X_FRACTION) * g.rail_x
        y = rng.uniform(-SERVE_Y_FRACTION, SERVE_Y_FRACTION) * g.puck_limit_y
        for skill in skills:
            skill.reset()
        return env.reset(puck_pos=(x, y), puck_vel=(0.0, 0.0)), side

    point = 0
    state, server = serve(point)
    point_start = 0.0
    for tick in range(round(seconds / env.step_seconds)):
        now = tick * env.step_seconds
        state = env.step(np.array([skill.target_one(state) for skill in skills]))
        qpos.append(env.data.qpos.copy())
        if state.goal:
            scorer = names[-1 if state.goal == "scored" else +1]
            score[scorer] += 1
            line = f"goal by {scorer}"
        elif now - point_start >= POINT_SECONDS:
            line = f"no goal ({names[server]} served)"
        else:
            continue
        tally = " - ".join(f"{name} {goals}" for name, goals in score.items())
        events.append((now, f"{line:32s} {tally}"))
        point += 1
        state, server = serve(point)
        point_start = now
    return np.array(qpos), events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("left", choices=SKILL_NAMES, help="skill for the left (robot) paddle")
    parser.add_argument("right", choices=SKILL_NAMES, help="skill for the right (opponent) paddle")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=None, help="default: outputs/showcase_<left>_vs_<right>.mp4"
    )
    args = parser.parse_args()

    env = AirHockeyEnv(layout=Layout.of(2, 1))
    qpos, events = play_match(env, args.left, args.right, args.seconds, args.seed)
    for time, line in events:
        print(f"{time:5.1f} s  {line}")

    from puckbot.rl.sim.video import write_video  # needs OpenGL

    out = args.out or Path("outputs") / f"showcase_{args.left}_vs_{args.right}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_video(env.model, SimpleNamespace(qpos=qpos, time=np.arange(len(qpos)) * env.step_seconds), out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
