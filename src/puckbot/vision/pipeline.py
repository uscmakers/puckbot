"""Analyse one photo: where are the table, the puck and the mallets?

Steps:
    1. Table    find the playing surface -> pixel-to-inch converter   (table_calibration.py)
    2. Objects  find the puck and mallets on it                       (puck_mallet_detector.py)
    3. Inches   convert their positions to inches
    4. Checks   list anything missing or suspicious, and how to fix it

Usage:
    result = analyze(image, load_settings())
    result.puck.position_in     # (x, y) inches
    result.mallets              # list
    result.problems             # [] if all good
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from puckbot.vision.puck_mallet_detector import Detections, PuckMalletDetector, Shape
from puckbot.vision.settings import Settings, TableSettings
from puckbot.vision.table_calibration import TableCalibration

# Real sizes in inches. Only used to sanity-check results.
PUCK_DIAMETER_IN = (2.5, 3.25)
MALLET_DIAMETER_IN = (3.0, 4.25)


# ── Output ────────────────────────────────────────────────────────────────────


@dataclass
class Located:
    """A puck or mallet: where it is in the photo, and on the table."""

    pixel: tuple[float, float]
    radius_px: float
    position_in: tuple[float, float] | None  # None if the table wasn't found
    diameter_in: float | None


@dataclass
class Analysis:
    table: TableCalibration | None  # None if the table wasn't found
    puck: Located | None
    mallets: list[Located]
    detections: Detections  # raw detector output, for debugging pictures
    problems: list[str]


# ── Pipeline ──────────────────────────────────────────────────────────────────


def analyze(image: np.ndarray, settings: Settings) -> Analysis:
    """Steps 1-4 on one photo (BGR, as loaded by cv2.imread)."""
    problems: list[str] = []

    # 1. Table
    table = _find_table(image, settings.table, problems)

    # 2. Objects
    outline = table.corners_px if table else None
    found = PuckMalletDetector(settings.detector).find(image, table_outline=outline)
    if not found.pucks:
        problems.append("Puck: not found. In glare, or touching a mallet/rail? Else tune 'puck' settings.")
    elif len(found.pucks) > 1:
        problems.append(f"Puck: {len(found.pucks)} puck-like shapes. Using the best one.")
    if not found.mallets:
        problems.append("Mallets: none on the table. (Mallets on the rails are ignored.)")

    # 3. Inches
    puck = _locate(found.puck, table) if found.puck else None
    mallets = [_locate(m, table) for m in found.mallets]

    # 4. Checks
    _check_size("Puck", puck, PUCK_DIAMETER_IN, problems)
    for mallet in mallets:
        _check_size("Mallet", mallet, MALLET_DIAMETER_IN, problems)

    return Analysis(table, puck, mallets, found, problems)


# ── Steps ─────────────────────────────────────────────────────────────────────


def _find_table(image: np.ndarray, settings: TableSettings, problems: list[str]) -> TableCalibration | None:
    try:
        table = TableCalibration.from_photo(image, settings)
    except ValueError as err:
        problems.append(f"Table: not found ({err}). Get the whole blue surface in the photo.")
        return None

    h, w = image.shape[:2]
    if any(min(u, v, w - u, h - v) < 10 for u, v in table.corners_px):
        problems.append("Table: a corner touches the photo's edge. Zoom out so all 4 are in frame.")
    return table


def _locate(shape: Shape, table: TableCalibration | None) -> Located:
    if table is None:
        return Located((shape.x, shape.y), shape.radius, None, None)
    x, y = table.to_table(shape.x, shape.y)
    edge_x, _ = table.to_table(shape.x + shape.radius, shape.y)
    return Located((shape.x, shape.y), shape.radius, (x, y), 2 * abs(edge_x - x))


def _check_size(name: str, item: Located | None, expected: tuple[float, float], problems: list[str]) -> None:
    """A size far from real life usually means the table size in the settings is wrong."""
    if item is None or item.diameter_in is None:
        return
    low, high = expected
    if not low * 0.8 <= item.diameter_in <= high * 1.2:
        problems.append(
            f"{name}: {item.diameter_in:.1f} in across, expected {low}-{high}. Check the table size setting."
        )
