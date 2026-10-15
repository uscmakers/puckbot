"""Find the puck and the mallets in a photo.

Both are white and round. So: find every white shape on the table once,
then sort each one into puck, mallet, or rejected (with the reason).

Steps:
    1. White mask  white = light, colourless pixels (puck, mallets, but also lines, logo, glare)
    2. Clean up    remove specks, fill small holes
    3. Table only  keep pixels inside the table outline (drops rails, floor, people)
    4. Shapes      trace the outline of each white region
    5. Measure     area, roundness, fill
    6. Sort        fits puck rules -> puck; fits mallet rules -> mallet; else rejected

Output is in pixels. table_calibration.py converts pixels to inches.

Usage:
    found = PuckMalletDetector(settings.detector).find(image, table_outline=cal.corners_px)
    found.puck      # best puck, or None
    found.mallets   # best first
    found.shapes    # every shape, incl. rejected ones and why
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from puckbot.vision.settings import DetectorSettings, ShapeRules

Kind = Literal["puck", "mallet", "rejected"]

# ── Output ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Shape:
    """One white region in the photo, measured. All sizes in pixels."""

    x: float  # centre, column
    y: float  # centre, row
    radius: float  # smallest circle around it
    area: float
    roundness: float  # 1.0 = circle, ~0.2 = long thin line
    fill: float  # 1.0 = solid disc, low = ring
    kind: Kind
    reason: str  # why it was rejected; "" otherwise
    score: float  # how well it fits its kind; higher = better
    outline: np.ndarray = field(repr=False, compare=False)


@dataclass
class Detections:
    pucks: list[Shape]  # best first
    mallets: list[Shape]  # best first
    shapes: list[Shape]  # everything found, incl. rejected

    @property
    def puck(self) -> Shape | None:
        return self.pucks[0] if self.pucks else None


# ── Detector ──────────────────────────────────────────────────────────────────


class PuckMalletDetector:
    def __init__(self, settings: DetectorSettings):
        self.settings = settings

    def find(self, image: np.ndarray, table_outline: np.ndarray | None = None) -> Detections:
        """Steps 1-6.

        table_outline: the table's corners in pixels. Without it, the whole photo is searched.
        """
        mask = self.white_mask(image, table_outline)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes = sorted((self._measure(c, mask) for c in contours), key=lambda s: -s.score)
        return Detections(
            pucks=[s for s in shapes if s.kind == "puck"],
            mallets=[s for s in shapes if s.kind == "mallet"],
            shapes=shapes,
        )

    def white_mask(self, image: np.ndarray, table_outline: np.ndarray | None = None) -> np.ndarray:
        """Steps 1-3. White = could be part of a puck or mallet."""
        s = self.settings

        # 1. White mask (blur first so camera noise doesn't make speckles)
        blurred = cv2.GaussianBlur(image, (s.blur_px | 1, s.blur_px | 1), 0)  # size must be odd
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(s.white_hsv_lower), np.array(s.white_hsv_upper))

        # 2. Clean up: "open" removes specks, "close" fills small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s.cleanup_px, s.cleanup_px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 3. Table only
        if table_outline is not None:
            on_table = np.zeros_like(mask)
            cv2.fillConvexPoly(on_table, np.round(table_outline).astype(np.int32), 255)
            mask &= on_table
        return mask

    def _measure(self, outline: np.ndarray, mask: np.ndarray) -> Shape:
        """Steps 5-6 for one shape."""
        # 5. Measure
        area = cv2.contourArea(outline)
        perimeter = cv2.arcLength(outline, True)
        roundness = 4 * np.pi * area / perimeter**2 if perimeter > 0 else 0.0  # circle = 1
        fill = _fill(mask, outline)
        (x, y), radius = cv2.minEnclosingCircle(outline)
        m = cv2.moments(outline)
        if m["m00"] > 0:  # centre of mass: steadier than the circle's centre
            x, y = m["m10"] / m["m00"], m["m01"] / m["m00"]

        # 6. Sort
        kind, reason, rules = self._sort(area, roundness, fill)
        score = roundness * _size_match(radius, rules.expected_radius_px) if kind != "rejected" else -1.0
        return Shape(x, y, radius, area, roundness, fill, kind, reason, score, outline)

    def _sort(self, area: float, roundness: float, fill: float) -> tuple[Kind, str, ShapeRules]:
        """-> (kind, reason if rejected, rules used)."""
        puck, mallet = self.settings.puck, self.settings.mallet
        puck_problem = _broken_rule(puck, area, roundness, fill)
        if not puck_problem:
            return "puck", "", puck
        mallet_problem = _broken_rule(mallet, area, roundness, fill)
        if not mallet_problem:
            return "mallet", "", mallet
        # Neither: give the reason from whichever size range it is nearer to
        if area <= puck.max_area_px:
            return "rejected", puck_problem, puck
        return "rejected", mallet_problem, mallet


# ── Helpers ───────────────────────────────────────────────────────────────────


def _broken_rule(rules: ShapeRules, area: float, roundness: float, fill: float) -> str:
    """First rule the shape breaks, or "" if it passes them all."""
    if area < rules.min_area_px:
        return "too small"
    if area > rules.max_area_px:
        return "too big"
    if roundness < rules.min_roundness:
        return "not round"
    if fill < rules.min_fill:
        return "ring, not solid"
    return ""


def _size_match(radius: float, expected: float) -> float:
    """1.0 at the expected radius, ~0.6 at half or double it."""
    return float(np.exp(-(np.log(radius / expected) ** 2) / 0.18))


def _fill(mask: np.ndarray, outline: np.ndarray) -> float:
    """Share of the area inside the outline that is white. Solid disc = 1.0, ring = low."""
    x, y, w, h = cv2.boundingRect(outline)
    inside = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(inside, [outline - (x, y)], -1, 255, thickness=cv2.FILLED)
    total = cv2.countNonZero(inside)
    white = cv2.countNonZero(mask[y : y + h, x : x + w] & inside)
    return white / total if total else 0.0
