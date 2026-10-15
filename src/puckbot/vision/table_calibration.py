"""Find the table in a photo, and convert pixels <-> positions on the table.

Steps:
    1. Blue mask      white = playing-surface pixels
    2. Outline        biggest blue region, with a shape wrapped around it
    3. Rough corners  simplify that shape to 4 points
    4. Exact corners  fit a straight line along each edge; intersect neighbours
    5. Homography     3x3 matrix: the 4 corners in pixels -> the 4 real corners

Step 5 is the key idea. Once the matrix is known it converts ANY pixel on the
table to a table position, correcting for the camera's angle.

Table positions are in inches:
    origin  surface corner at the top-left of the photo
    x       across the table  (0 .. width)
    y       along the table   (0 .. length)

Usage:
    cal = TableCalibration.from_photo(image, settings.table)
    x, y = cal.to_table(u, v)      # pixel -> inches
    view = cal.top_down(image)     # straightened view of the table

Known limit: from an angle the camera also sees the inside walls of the rails.
They are blue too, so an edge can be ~0.5 in off. A camera straight overhead helps.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from puckbot.vision.settings import TableSettings

# ── Calibration ───────────────────────────────────────────────────────────────


@dataclass
class TableCalibration:
    corners_px: np.ndarray  # surface corners in the photo: TL, TR, BR, BL
    size_in: tuple[float, float]  # surface (width, length)
    homography: np.ndarray  # 3x3 matrix: pixel -> inches

    @classmethod
    def from_photo(cls, image: np.ndarray, table: TableSettings) -> TableCalibration:
        """Steps 1-5. Raises ValueError if the table can't be found."""
        corners = find_surface_corners(image, table)
        width, length = table.surface_size_in

        # x runs along the short side. If the table lies sideways in the photo, rotate to match.
        top_edge = np.linalg.norm(corners[1] - corners[0])
        left_edge = np.linalg.norm(corners[3] - corners[0])
        if top_edge <= left_edge:
            real = [(0, 0), (width, 0), (width, length), (0, length)]
        else:
            real = [(0, 0), (0, length), (width, length), (width, 0)]

        homography, _ = cv2.findHomography(corners, np.array(real, dtype=np.float64))
        if homography is None:
            raise ValueError("table corners are in a line")
        return cls(corners, (width, length), homography)

    def to_table(self, u: float, v: float) -> tuple[float, float]:
        """Pixel (column u, row v) -> table position (x, y) in inches."""
        return _transform(self.homography, u, v)

    def to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """Table position (x, y) in inches -> pixel (column, row)."""
        return _transform(np.linalg.inv(self.homography), x, y)

    def top_down(self, image: np.ndarray, px_per_in: float = 8.0, margin_in: float = 2.5) -> np.ndarray:
        """Straightened view of the table, as if photographed from straight above.

        Output pixel (i, j) shows table position (i / px_per_in - margin_in, j / px_per_in - margin_in).
        """
        shift = np.array([[1, 0, margin_in], [0, 1, margin_in], [0, 0, 1]], dtype=np.float64)
        zoom = np.diag([px_per_in, px_per_in, 1.0])
        width, length = self.size_in
        size = (round((width + 2 * margin_in) * px_per_in), round((length + 2 * margin_in) * px_per_in))
        return cv2.warpPerspective(image, zoom @ shift @ self.homography, size)


def _transform(matrix: np.ndarray, a: float, b: float) -> tuple[float, float]:
    x, y = cv2.perspectiveTransform(np.array([[[a, b]]], dtype=np.float64), matrix)[0, 0]
    return float(x), float(y)


# ── Finding the corners ───────────────────────────────────────────────────────


def surface_mask(image: np.ndarray, table: TableSettings) -> np.ndarray:
    """Step 1. White = playing-surface colour."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(table.surface_hsv_lower), np.array(table.surface_hsv_upper))


def find_surface_corners(image: np.ndarray, table: TableSettings) -> np.ndarray:
    """Steps 1-4. The playing surface's corners in pixels: TL, TR, BR, BL."""
    # 1. Blue mask ("close" fills small gaps, e.g. thin lines on the table)
    mask = surface_mask(image, table)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    # 2. Outline of the biggest blue region, wrapped (convex hull) to bridge the puck/mallet holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("no table colour in the photo")
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))

    # 3. Rough corners
    rough = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    if len(rough) != 4:
        raise ValueError(f"table outline has {len(rough)} corners, expected 4")
    rough = _sort_corners(rough.astype(np.float64))

    # 4. Exact corners: rounded corners and mallets on the edge would pull rough corners off,
    #    so fit a straight line to the middle of each edge and intersect neighbouring lines.
    outline = _add_points_along(hull.reshape(-1, 2).astype(np.float64))
    edges = [_fit_edge(outline, rough[i], rough[(i + 1) % 4]) for i in range(4)]
    return np.array([_intersect(edges[i - 1], edges[i]) for i in range(4)])


def _sort_corners(points: np.ndarray) -> np.ndarray:
    """4 points -> TL, TR, BR, BL."""
    total = points.sum(axis=1)  # x + y: smallest at TL, largest at BR
    diff = points[:, 0] - points[:, 1]  # x - y: largest at TR, smallest at BL
    return points[[np.argmin(total), np.argmax(diff), np.argmax(total), np.argmin(diff)]]


def _add_points_along(outline: np.ndarray, step_px: float = 2.0) -> np.ndarray:
    """A hull has few points (one per bend). Add one every 2 px so each edge has plenty to fit."""
    out = []
    for p, q in zip(outline, np.roll(outline, -1, axis=0), strict=True):
        n = max(int(np.linalg.norm(q - p) / step_px), 1)
        out.append(p + (q - p) * np.linspace(0, 1, n, endpoint=False)[:, None])
    return np.vstack(out)


def _fit_edge(outline: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Straight line along edge a -> b, as (point, direction).

    Uses only outline points in the middle 70% of the edge (skips rounded corners)
    and within 6 px of it (skips bumps).
    """
    ab = b - a
    length = np.linalg.norm(ab)
    along = (outline - a) @ ab / length**2  # 0 at a, 1 at b
    away = np.abs((outline - a) @ np.array([-ab[1], ab[0]]) / length)  # distance from the edge
    near = outline[(along > 0.15) & (along < 0.85) & (away < 6)]
    if len(near) < 2:
        return a, ab / length
    vx, vy, x0, y0 = cv2.fitLine(near.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    return np.array([x0, y0]), np.array([vx, vy])


def _intersect(line1, line2) -> np.ndarray:
    """Where two lines (point, direction) cross."""
    (p, r), (q, s) = line1, line2
    cross = r[0] * s[1] - r[1] * s[0]
    if abs(cross) < 1e-9:
        raise ValueError("table edges are parallel")
    t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / cross
    return p + t * r
