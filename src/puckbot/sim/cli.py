"""Command line: run hand-written skills in the sim.

Usage:
    puckbot-sim --list                          show the drills
    puckbot-sim                                 run every drill, print the outcome of each
    puckbot-sim bank_shot goalie                run just these
    puckbot-sim --out-dir outputs               + save each run (.npz) there
    puckbot-sim --out-dir outputs --video       + save an mp4 of each (needs MUJOCO_GL=egl, i.e. a GPU node)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from puckbot.sim.drills import DRILLS, run_drill
from puckbot.sim.env import AirHockeyEnv
from puckbot.sim.settings import DEFAULT_SETTINGS, load_settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("drills", nargs="*", metavar="drill", help="which drills (default: all)")
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
    if args.video and not args.out_dir:
        parser.error("--video needs --out-dir")

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
            from puckbot.sim.video import write_video  # needs OpenGL, so only load it when asked

            write_video(env.model, run, out_dir / f"{name}.mp4")


if __name__ == "__main__":
    main()
