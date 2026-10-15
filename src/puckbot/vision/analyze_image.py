"""Analyse one overhead photo of the table: find the table, the puck and the mallets.

    puckbot-analyze data/samples/sample_hockey_table_top_view.png
    puckbot-analyze my_photo.jpg --out outputs/

Steps (each prints what it found, or what is missing and how to fix it):
  1. Table   - find the blue playing surface and its 4 corners -> pixel <-> inch mapping.
               Surface size = outer table size minus a rail on each side (configs/our_table.json).
  2. Puck    - solid white disc, puck-sized, on the blue surface.
  3. Mallets - larger white round blobs on the blue surface.
  4. Report  - positions in inches. Origin = playing-surface corner at the top-left of the
               photo; x runs across the table, y along it.

With --out it saves <name>_analyzed.png: the photo with everything drawn on it, next to a
straightened top-down view of the table with a 10-inch grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from puckbot.vision.calibration import TableCalibration
from puckbot.vision.puck_detector import PuckDetection, PuckDetector, PuckDetectorConfig

MM_PER_IN = 25.4
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "our_table.json"

# Real sizes, used only for sanity checks (typical full-size table gear).
PUCK_DIAMETER_IN = (2.5, 3.25)
MALLET_DIAMETER_IN = (3.0, 4.25)

GREEN, ORANGE, YELLOW, RED = (0, 255, 0), (0, 140, 255), (0, 255, 255), (0, 0, 255)


def analyze(image: np.ndarray, config: dict) -> dict:
    """Run the whole pipeline on one BGR image. Returns positions and a list of problems."""
    table = config["table"]
    width_in = table["outer_width_in"] - 2 * table["rail_width_in"]
    length_in = table["outer_length_in"] - 2 * table["rail_width_in"]
    problems: list[str] = []

    # 1. Table
    cal = None
    try:
        cal = TableCalibration.from_surface(
            image,
            tuple(table["surface_hsv_lower"]),
            tuple(table["surface_hsv_upper"]),
            width_in * MM_PER_IN,
            length_in * MM_PER_IN,
        )
    except ValueError as err:
        problems.append(
            f"Table: could not find the 4 corners of the blue surface ({err}). Make sure the whole "
            "blue surface is in the photo and evenly lit, or adjust surface_hsv_* in the config."
        )
    if cal is not None:
        h, w = image.shape[:2]
        near_edge = [(u, v) for u, v in cal.image_points if min(u, v, w - u, h - v) < 10]
        if near_edge:
            problems.append(
                "Table: a surface corner is at the edge of the photo, so the table may be cut off. "
                "Zoom out or raise the camera so all four corners are in frame."
            )

    # 2 + 3. Puck and mallets
    pucks = PuckDetector(PuckDetectorConfig.from_dict(config["puck"])).candidates(image)
    mallets = PuckDetector(PuckDetectorConfig.from_dict(config["mallet"])).candidates(image)
    if not pucks:
        problems.append(
            "Puck: none found on the blue surface. Check it isn't touching a mallet or the rail, "
            "and that it isn't in glare. Tune the 'puck' section of the config if needed."
        )
    elif len(pucks) > 1:
        problems.append(f"Puck: {len(pucks)} puck-like blobs found; using the best one.")
    if not mallets:
        problems.append("Mallets: none found on the blue surface (mallets on the rails are ignored).")

    # 4. Convert to inches and sanity-check sizes
    def place(det: PuckDetection) -> dict:
        out = {"px": (det.x, det.y), "radius_px": det.radius}
        if cal is not None:
            x, y = cal.px_to_mm(det.x, det.y)
            out["in"] = (x / MM_PER_IN, y / MM_PER_IN)
            edge_x, _ = cal.px_to_mm(det.x + det.radius, det.y)
            out["diameter_in"] = 2 * abs(edge_x - x) / MM_PER_IN
        return out

    result = {
        "calibration": cal,
        "surface_in": (width_in, length_in),
        "puck": place(pucks[0]) if pucks else None,
        "mallets": [place(m) for m in mallets],
        "problems": problems,
    }
    for name, item, (lo, hi) in [("Puck", result["puck"], PUCK_DIAMETER_IN)] + [
        ("Mallet", m, MALLET_DIAMETER_IN) for m in result["mallets"]
    ]:
        if item and "diameter_in" in item and not lo * 0.8 <= item["diameter_in"] <= hi * 1.2:
            problems.append(
                f"{name}: measures {item['diameter_in']:.1f} in across (expected {lo}-{hi} in). "
                "The table size in the config may be wrong; measure the blue surface inside the rails."
            )
    return result


def draw(image: np.ndarray, result: dict) -> np.ndarray:
    """Annotated photo next to a top-down view of the table."""
    view = image.copy()
    cal: TableCalibration | None = result["calibration"]
    if cal is not None:
        cv2.polylines(view, [cal.image_points.astype(np.int32)], True, YELLOW, 2)
    for item, color, label in [(result["puck"], GREEN, "puck")] + [
        (m, ORANGE, "mallet") for m in result["mallets"]
    ]:
        if item is None:
            continue
        u, v = (int(round(c)) for c in item["px"])
        cv2.circle(view, (u, v), int(round(item["radius_px"])), color, 3)
        text = label if "in" not in item else f"{label} ({item['in'][0]:.1f}, {item['in'][1]:.1f}) in"
        cv2.putText(view, text, (u + 12, v - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if cal is None:
        return view

    px_per_mm, margin = 0.3, 60.0
    top = cal.warp(image, px_per_mm=px_per_mm, margin_mm=margin)

    def to_px(x_mm, y_mm):
        return int(round((x_mm + margin) * px_per_mm)), int(round((y_mm + margin) * px_per_mm))

    step = 10 * MM_PER_IN
    for x in np.arange(0, cal.width_mm + 1, step):
        cv2.line(top, to_px(x, 0), to_px(x, cal.length_mm), (200, 200, 200), 1)
    for y in np.arange(0, cal.length_mm + 1, step):
        cv2.line(top, to_px(0, y), to_px(cal.width_mm, y), (200, 200, 200), 1)
    cv2.rectangle(top, to_px(0, 0), to_px(cal.width_mm, cal.length_mm), YELLOW, 2)
    for item, color in [(result["puck"], GREEN)] + [(m, ORANGE) for m in result["mallets"]]:
        if item is not None and "in" in item:
            x_in, y_in = item["in"]
            cv2.circle(top, to_px(x_in * MM_PER_IN, y_in * MM_PER_IN), 10, color, 3)

    scale = view.shape[0] / top.shape[0]
    top = cv2.resize(top, (int(top.shape[1] * scale), view.shape[0]))
    return np.hstack([view, top])


def print_report(path: str, result: dict) -> None:
    w, l = result["surface_in"]
    print(f"{path}")
    if result["calibration"] is not None:
        print(f"  table   : found, playing surface {w:.1f} x {l:.1f} in")
    else:
        print("  table   : NOT found (positions in pixels only)")

    def fmt(item):
        s = f"pixel ({item['px'][0]:.0f}, {item['px'][1]:.0f})"
        if "in" in item:
            s += f" -> x={item['in'][0]:.1f} in, y={item['in'][1]:.1f} in, ~{item['diameter_in']:.1f} in across"
        return s

    print(f"  puck    : {fmt(result['puck']) if result['puck'] else 'not found'}")
    if result["mallets"]:
        for i, m in enumerate(result["mallets"], 1):
            print(f"  mallet {i}: {fmt(m)}")
    else:
        print("  mallets : none on the table")
    if result["problems"]:
        print("  missing / check:")
        for p in result["problems"]:
            print(f"    - {p}")
    else:
        print("  all good")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="overhead photo of the table")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="table / puck / mallet settings")
    p.add_argument("--out", help="folder for the annotated image")
    args = p.parse_args(argv)

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"could not read image {args.image}")
    result = analyze(image, json.loads(Path(args.config).read_text()))
    print_report(args.image, result)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{Path(args.image).stem}_analyzed.png"
        cv2.imwrite(str(path), draw(image, result))
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
