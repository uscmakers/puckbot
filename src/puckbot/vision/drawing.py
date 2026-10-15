"""Pictures of an analysis, for people to look at.

Usage:
    draw_result(image, analysis)            photo with the table, puck and mallets marked
    draw_top_down(image, analysis)          straightened table with a 10-inch grid
    draw_steps(image, settings, analysis)   every pipeline step in one numbered grid
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from puckbot.vision.pipeline import Analysis
from puckbot.vision.puck_mallet_detector import PuckMalletDetector
from puckbot.vision.settings import Settings
from puckbot.vision.table_calibration import surface_mask

# Colours are (Blue, Green, Red): OpenCV's order.
GREEN = (0, 255, 0)  # puck
ORANGE = (0, 140, 255)  # mallets
YELLOW = (0, 255, 255)  # table
RED = (0, 0, 255)  # rejected shapes
WHITE = (255, 255, 255)
GREY = (200, 200, 200)

KIND_COLOUR = {"puck": GREEN, "mallet": ORANGE, "rejected": RED}


# ── Pictures ──────────────────────────────────────────────────────────────────


def draw_result(image: np.ndarray, a: Analysis, units: Literal["in", "px"] = "in") -> np.ndarray:
    """Photo with the table outline (yellow), puck (green) and mallets (orange).

    units: label each one with its table position ("in") or its pixel position ("px").
    """
    view = image.copy()
    if a.table:
        cv2.polylines(view, [a.table.corners_px.astype(np.int32)], True, YELLOW, 3)

    for item, colour, name in [(a.puck, GREEN, "puck")] + [(m, ORANGE, "mallet") for m in a.mallets]:
        if item is None:
            continue
        u, v = round(item.pixel[0]), round(item.pixel[1])
        cv2.circle(view, (u, v), round(item.radius_px), colour, 4)
        if units == "px":
            name += f" ({u}, {v}) px r={item.radius_px:.0f}"
        elif item.position_in:
            name += f" ({item.position_in[0]:.1f}, {item.position_in[1]:.1f}) in"
        _label(view, name, (u + 15, v - 15), colour, 0.9)
    return view


def draw_top_down(
    image: np.ndarray, a: Analysis, px_per_in: float = 8.0, margin_in: float = 2.5
) -> np.ndarray | None:
    """Straightened view of the table with a 10-inch grid. None if no table was found."""
    if a.table is None:
        return None
    view = a.table.top_down(image, px_per_in, margin_in)
    width, length = a.table.size_in

    def at(x_in: float, y_in: float) -> tuple[int, int]:
        return round((x_in + margin_in) * px_per_in), round((y_in + margin_in) * px_per_in)

    for x in np.arange(0, width + 0.1, 10):
        cv2.line(view, at(x, 0), at(x, length), GREY, 1)
    for y in np.arange(0, length + 0.1, 10):
        cv2.line(view, at(0, y), at(width, y), GREY, 1)
    cv2.rectangle(view, at(0, 0), at(width, length), YELLOW, 2)

    for item, colour in [(a.puck, GREEN)] + [(m, ORANGE) for m in a.mallets]:
        if item and item.position_in:
            x, y = item.position_in
            cv2.circle(view, at(x, y), 10, colour, 3)
            _label(view, f"({x:.1f}, {y:.1f}) in", (at(x, y)[0] + 14, at(x, y)[1] + 5), colour, 0.5)
    return view


def draw_steps(image: np.ndarray, settings: Settings, a: Analysis) -> np.ndarray:
    """Every pipeline step, numbered, in one grid."""
    detector = PuckMalletDetector(settings.detector)
    outline = a.table.corners_px if a.table else None

    tiles = [
        ("0. Original photo", image),
        ("1a. Blue = table", surface_mask(image, settings.table)),
        ("1b. Table corners", _draw_corners(image, a)),
        ("2a. White, anywhere", detector.white_mask(image)),
        ("2b. White, on table", detector.white_mask(image, outline)),
        ("2c. Classify each entity", _draw_shapes(image, a)),
        ("3. Get puck/mallet pixel info", draw_result(image, a, units="px")),
        ("4. Get puck/mallet coords on table", draw_top_down(image, a)),
    ]
    return grid([(title, img) for title, img in tiles if img is not None], columns=4)


# ── Step tiles ────────────────────────────────────────────────────────────────


def _draw_corners(image: np.ndarray, a: Analysis) -> np.ndarray:
    view = image.copy()
    if a.table:
        corners = a.table.corners_px
        cv2.polylines(view, [corners.astype(np.int32)], True, YELLOW, 4)
        for (u, v), name in zip(corners, ["TL", "TR", "BR", "BL"], strict=True):
            cv2.circle(view, (round(u), round(v)), 14, YELLOW, -1)
            _label(view, name, (round(u) + 18, round(v) + 40), YELLOW, 1.2)
    return view


def _draw_shapes(image: np.ndarray, a: Analysis) -> np.ndarray:
    """Each white shape's outline: green = puck, orange = mallet, red = rejected (+ why)."""
    view = image.copy()
    for shape in a.detections.shapes:
        if shape.area < 100:  # specks: not worth drawing
            continue
        colour = KIND_COLOUR[shape.kind]
        cv2.drawContours(view, [shape.outline], -1, colour, 4)
        if shape.kind != "rejected" or shape.area >= 1000:  # label only bigger ones: stays readable
            x, y, _, _ = cv2.boundingRect(shape.outline)
            text = shape.kind.upper() if shape.kind != "rejected" else f"no: {shape.reason}"
            _label(view, text, (x, max(y - 12, 30)), colour, 1.1)
    return view


# ── Layout helpers ────────────────────────────────────────────────────────────


def grid(tiles: list[tuple[str, np.ndarray]], columns: int, tile_width: int = 420) -> np.ndarray:
    """(title, image) tiles laid out left to right, top to bottom, each under a title bar."""
    tile_height = round(tile_width * 1.6)  # tall: the table is portrait in our photos
    bar = 44
    cells = []
    for title, img in tiles:
        if img.ndim == 2:  # black/white mask -> colour, so it can sit next to photos
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        scale = min(tile_width / img.shape[1], tile_height / img.shape[0])
        img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)))

        cell = np.full((tile_height + bar, tile_width, 3), 30, dtype=np.uint8)  # dark grey
        top = bar + (tile_height - img.shape[0]) // 2
        left = (tile_width - img.shape[1]) // 2
        cell[top : top + img.shape[0], left : left + img.shape[1]] = img
        _label(cell, title, (10, 31), WHITE, _fit_text(title, tile_width - 20, 0.8))
        cells.append(cell)

    while len(cells) % columns:  # pad the last row
        cells.append(np.full_like(cells[0], 30))
    rows = [np.hstack(cells[i : i + columns]) for i in range(0, len(cells), columns)]
    return np.vstack(rows)


def _label(img: np.ndarray, text: str, at: tuple[int, int], colour, size: float) -> None:
    """Text with a black outline, so it reads on any background."""
    thickness = max(2, round(size * 2))
    cv2.putText(img, text, at, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, at, cv2.FONT_HERSHEY_SIMPLEX, size, colour, thickness, cv2.LINE_AA)


def _fit_text(text: str, max_width: int, size: float) -> float:
    """Largest font size (up to `size`) at which `text` fits in `max_width` pixels."""
    while size > 0.3:
        thickness = max(2, round(size * 2)) + 3  # outline included
        (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thickness)
        if width <= max_width:
            break
        size -= 0.05
    return size
