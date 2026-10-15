"""Table calibration: convert image pixels to table millimetres.

Table frame (all lengths in mm):
    origin = one corner of the playing surface, x = across the table (0..width),
    y = along the table (0..length). Which end is "ours" is decided by the caller.

Calibrate from the playing surface itself (nothing to stick on the table):
       cal = TableCalibration.from_surface(image, table_hsv_lower, table_hsv_upper, width_mm, length_mm)
   Finds the coloured surface, fits a straight line to each of its four edges and intersects them.
   Needs the INSIDE-the-rails surface size. Any 4+ known points also work: TableCalibration.from_points.

Known limit: from an angle the camera also sees the inner walls of the near rails, which are the
same colour, so those edges can land ~1-2 cm out. A camera straight above the centre minimises it.

Then:
       X, Y = cal.px_to_mm(u, v)
       top_down = cal.warp(image, px_per_mm=0.5)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

@dataclass
class TableCalibration:
    homography: np.ndarray  # 3x3, image pixels -> table mm
    width_mm: float
    length_mm: float
    image_points: np.ndarray | None = None  # pixel points used to fit (N x 2), for drawing

    # ---- conversions -------------------------------------------------------------------------

    def px_to_mm(self, u: float, v: float) -> tuple[float, float]:
        x, y = cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float64), self.homography)[0, 0]
        return float(x), float(y)

    def mm_to_px(self, x: float, y: float) -> tuple[float, float]:
        inv = np.linalg.inv(self.homography)
        u, v = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float64), inv)[0, 0]
        return float(u), float(v)

    def warp(self, image: np.ndarray, px_per_mm: float = 0.5, margin_mm: float = 0.0) -> np.ndarray:
        """Top-down view of the table: 1 output pixel = 1/px_per_mm mm, origin at the top left."""
        shift = np.array([[1, 0, margin_mm], [0, 1, margin_mm], [0, 0, 1]], dtype=np.float64)
        scale = np.diag([px_per_mm, px_per_mm, 1.0])
        size = (
            int(round((self.width_mm + 2 * margin_mm) * px_per_mm)),
            int(round((self.length_mm + 2 * margin_mm) * px_per_mm)),
        )
        return cv2.warpPerspective(image, scale @ shift @ self.homography, size)

    # ---- construction ------------------------------------------------------------------------

    @classmethod
    def from_points(
        cls, image_points: np.ndarray, table_points_mm: np.ndarray, width_mm: float, length_mm: float
    ) -> "TableCalibration":
        image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        table_points_mm = np.asarray(table_points_mm, dtype=np.float64).reshape(-1, 2)
        if len(image_points) < 4:
            raise ValueError("need at least 4 point pairs")
        h, _ = cv2.findHomography(image_points, table_points_mm, method=0)
        if h is None:
            raise ValueError("could not fit a homography (points degenerate?)")
        return cls(h, width_mm, length_mm, image_points)

    @classmethod
    def from_surface(
        cls,
        image: np.ndarray,
        table_hsv_lower: tuple[int, int, int],
        table_hsv_upper: tuple[int, int, int],
        width_mm: float,
        length_mm: float,
    ) -> "TableCalibration":
        """Calibrate from the four corners of the coloured playing surface.

        The short edge of the surface becomes x (width), the long edge y (length).
        """
        corners = find_surface_corners(image, table_hsv_lower, table_hsv_upper)  # TL, TR, BR, BL
        top = np.linalg.norm(corners[1] - corners[0])
        side = np.linalg.norm(corners[3] - corners[0])
        if top <= side:  # TL->TR is the short edge
            mm = [(0, 0), (width_mm, 0), (width_mm, length_mm), (0, length_mm)]
        else:  # TL->TR is the long edge: rotate so x still runs across the short edge
            mm = [(0, 0), (0, length_mm), (width_mm, length_mm), (width_mm, 0)]
        return cls.from_points(corners, np.array(mm, dtype=np.float64), width_mm, length_mm)


    # ---- persistence -------------------------------------------------------------------------

    def to_json(self, path: str | Path) -> None:
        data = {
            "homography": self.homography.tolist(),
            "width_mm": self.width_mm,
            "length_mm": self.length_mm,
            "image_points": None if self.image_points is None else self.image_points.tolist(),
        }
        Path(path).write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def from_json(cls, path: str | Path) -> "TableCalibration":
        d = json.loads(Path(path).read_text())
        pts = None if d.get("image_points") is None else np.array(d["image_points"], dtype=np.float64)
        return cls(np.array(d["homography"], dtype=np.float64), d["width_mm"], d["length_mm"], pts)


# ---- helpers -----------------------------------------------------------------------------------


def find_surface_corners(image: np.ndarray, hsv_lower, hsv_upper) -> np.ndarray:
    """Four corners of the playing surface (TL, TR, BR, BL in image order), sub-pixel.

    Rounded corners and things lying on the edge (mallets) are handled by fitting a straight line
    to the middle part of each edge and intersecting neighbouring lines.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("table colour not found")
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2).astype(np.float64)
    if len(quad) != 4:
        raise ValueError(f"surface outline has {len(quad)} corners, expected 4")
    quad = _order_corners(quad)

    pts = hull.reshape(-1, 2).astype(np.float64)
    dense = _densify(pts)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        lines.append(_fit_edge(dense, a, b))
    corners = np.array([_intersect(lines[i - 1], lines[i]) for i in range(4)])
    return corners


def _order_corners(q: np.ndarray) -> np.ndarray:
    """Return TL, TR, BR, BL."""
    s, d = q.sum(axis=1), q[:, 0] - q[:, 1]
    return np.array([q[np.argmin(s)], q[np.argmax(d)], q[np.argmax(s)], q[np.argmin(d)]])


def _densify(pts: np.ndarray, step: float = 2.0) -> np.ndarray:
    out = []
    for p, q in zip(pts, np.roll(pts, -1, axis=0)):
        n = max(int(np.linalg.norm(q - p) / step), 1)
        out.append(p + (q - p) * np.linspace(0, 1, n, endpoint=False)[:, None])
    return np.vstack(out)


def _fit_edge(pts: np.ndarray, a: np.ndarray, b: np.ndarray, trim: float = 0.15, tol: float = 6.0):
    """Line (point, direction) through the hull points near segment a-b, ignoring the rounded ends."""
    ab = b - a
    length = np.linalg.norm(ab)
    t = (pts - a) @ ab / (length * length)
    normal = np.array([-ab[1], ab[0]]) / length
    dist = np.abs((pts - a) @ normal)
    sel = pts[(t > trim) & (t < 1 - trim) & (dist < tol)]
    if len(sel) < 2:
        return a, ab / length
    vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    return np.array([x0, y0], dtype=np.float64), np.array([vx, vy], dtype=np.float64)


def _intersect(l1, l2) -> np.ndarray:
    (p, r), (q, s) = l1, l2
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-9:
        raise ValueError("parallel edges")
    t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / denom
    return p + t * r
