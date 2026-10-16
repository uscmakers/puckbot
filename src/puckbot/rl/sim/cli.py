"""Command line: run hand-written skills in the sim.

Usage:
    puckbot-sim --list                          show the drills
    puckbot-sim                                 run every drill, print the outcome of each
    puckbot-sim bank_shot goalie                run just these
    puckbot-sim --out-dir outputs               + save each run (.npz) there
    puckbot-sim straight_shot bank_shot --grid  shoot from a grid of start spots (empty goal), print a map
    puckbot-sim --grid --goalie                 the same with the scripted goalie defending the far goal
    puckbot-sim --out-dir outputs --video       + save an mp4 of each (needs MUJOCO_GL=egl, i.e. a GPU node)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from puckbot.rl.sim.baselines import shot_grid
from puckbot.rl.sim.drills import DRILLS, run_drill
from puckbot.rl.sim.env import AirHockeyEnv
from puckbot.rl.sim.layout import Layout
from puckbot.rl.sim.settings import DEFAULT_SETTINGS, load_settings

SHOT_SKILLS = ("straight_shot", "bank_shot")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("drills", nargs="*", metavar="drill", help="which drills (default: all)")
    parser.add_argument(
        "--grid", action="store_true", help="shot skills only: score rate from a grid of starts"
    )
    parser.add_argument(
        "--goalie", action="store_true", help="with --grid: a scripted goalie defends the far goal"
    )
    parser.add_argument("--list", action="store_true", help="show the drills and exit")
    parser.add_argument("--settings", default=str(DEFAULT_SETTINGS), help="settings file (JSON)")
    parser.add_argument("--out-dir", help="save each run here as <drill>.npz")
    parser.add_argument("--video", action="store_true", help="also save <drill>.mp4 with MuJoCo's renderer")
    args = parser.parse_args(argv)

    if args.list:
        for name, drill in DRILLS.items():
            print(f"{name:14s} {drill.about}")
        return
    unknown = [name for name in args.drills if name not in DRILLS]
    if unknown:
        parser.error(f"unknown drill {unknown}; choose from {list(DRILLS)}")
    if args.goalie and not args.grid:
        parser.error("--goalie goes with --grid")
    if args.grid and set(args.drills) - set(SHOT_SKILLS):
        parser.error(f"--grid works with {list(SHOT_SKILLS)} only")
    if args.video and not args.out_dir:
        parser.error("--video needs --out-dir")

    if args.grid:
        env = AirHockeyEnv(load_settings(args.settings), Layout.of(2 if args.goalie else 1))
        for name in args.drills or SHOT_SKILLS:
            print(shot_grid(env, DRILLS[name].skill, opponent="goalie" if args.goalie else None), end="\n\n")
        return
    env = AirHockeyEnv(load_settings(args.settings))
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.drills or DRILLS:
        run = run_drill(env, name)
        print(
            f"{name:14s} {run.outcome:9s} {run.time[-1]:4.1f} s   top puck speed {run.top_puck_speed:.1f} m/s"
        )
        if out_dir:
            run.save(out_dir / f"{name}.npz")
        if args.video:
            from puckbot.rl.sim.video import write_video  # needs OpenGL, so only load it when asked

            write_video(env.model, run, out_dir / f"{name}.mp4")


if __name__ == "__main__":
    main()
