"""Command line: analyse a photo of the table.

Usage:
    puckbot-analyze photo.png                     print where everything is
    puckbot-analyze photo.png --out result.png    + save the marked-up photo and top-down view
    puckbot-analyze photo.png --steps steps.png   + save every pipeline step in one picture
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from puckbot.vision.drawing import draw_result, draw_steps, draw_top_down, grid
from puckbot.vision.pipeline import Analysis, Located, analyze
from puckbot.vision.settings import DEFAULT_SETTINGS, load_settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("photo", help="overhead photo of the table")
    parser.add_argument("--settings", default=str(DEFAULT_SETTINGS), help="settings file (JSON)")
    parser.add_argument("--out", help="save the marked-up photo + top-down view here (.png)")
    parser.add_argument("--steps", help="save every pipeline step in one picture here (.png)")
    args = parser.parse_args(argv)

    image = cv2.imread(args.photo)
    if image is None:
        raise SystemExit(f"could not read {args.photo}")
    settings = load_settings(args.settings)

    analysis = analyze(image, settings)
    print(report(args.photo, analysis))

    if args.out:
        top_down = draw_top_down(image, analysis)
        tiles = [("Result", draw_result(image, analysis))]
        if top_down is not None:
            tiles.append(("Top-down, 10 in grid", top_down))
        _save(args.out, grid(tiles, columns=len(tiles), tile_width=700))
    if args.steps:
        _save(args.steps, draw_steps(image, settings, analysis))


def report(name: str, a: Analysis) -> str:
    """Plain-text summary: what was found, and what is missing."""
    lines = [name]
    if a.table:
        width, length = a.table.size_in
        lines.append(f"  table   : found, playing surface {width:.1f} x {length:.1f} in")
    else:
        lines.append("  table   : NOT found (positions in pixels only)")

    lines.append(f"  puck    : {_describe(a.puck) if a.puck else 'not found'}")
    for i, mallet in enumerate(a.mallets, 1):
        lines.append(f"  mallet {i}: {_describe(mallet)}")
    if not a.mallets:
        lines.append("  mallets : none on the table")

    if a.problems:
        lines.append("  missing / check:")
        lines += [f"    - {p}" for p in a.problems]
    else:
        lines.append("  all good")
    return "\n".join(lines)


def _describe(item: Located) -> str:
    text = f"pixel ({item.pixel[0]:.0f}, {item.pixel[1]:.0f})"
    if item.position_in:
        x, y = item.position_in
        text += f" -> x={x:.1f} in, y={y:.1f} in, ~{item.diameter_in:.1f} in across"
    return text


def _save(path: str, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, img)
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
