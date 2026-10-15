"""Puck detection: one image in, puck pixel position out.

    detector = PuckDetector(PuckDetectorConfig(mode="hsv", hsv_lower=(0, 120, 70), hsv_upper=(10, 255, 255)))
    detection = detector.detect(image)        # PuckDetection or None
    if detection:
        print(detection.x, detection.y)

The detector is stateless except for an optional hint: pass `previous=(x, y)` to prefer the
candidate nearest the last known position when more than one blob looks like a puck.

Masking modes (pick with `PuckDetectorConfig.mode`):
    "hsv"    - colour range in HSV. Best for a coloured puck (red, yellow, green...).
    "bright" - grayscale threshold, pixels ABOVE `threshold`. For retroreflective tape + a light.
    "dark"   - grayscale threshold, pixels BELOW `threshold`. For a black puck on a light table.

Use `puckbot-pick-color` on a frame to get HSV numbers for your puck.
Pixels only: converting to table millimetres is the calibration step's job.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np

Mode = Literal["hsv", "bright", "dark"]


@dataclass
class PuckDetectorConfig:
    mode: Mode = "hsv"
    # HSV range (OpenCV: H 0-179, S 0-255, V 0-255). Default is a saturated red.
    hsv_lower: tuple[int, int, int] = (0, 120, 70)
    hsv_upper: tuple[int, int, int] = (10, 255, 255)
    # Red wraps around H=180; set this to also accept a second range, e.g. ((170,120,70),(179,255,255)).
    hsv_extra_range: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
    # Grayscale threshold for "bright" / "dark" modes.
    threshold: int = 200
    # Expected puck radius in pixels. None = accept any size within the area limits.
    expected_radius_px: float | None = None
    min_area_px: float = 30.0
    max_area_px: float = 50_000.0
    # 1.0 = perfect circle. Motion blur stretches the puck, so keep this lenient.
    min_circularity: float = 0.45
    # Fraction of the blob's outline that is filled. Rejects rings (mallet tops, centre circle).
    min_fill: float = 0.0
    # Playing-surface colour (HSV). When set, only blobs inside the surface (its convex hull)
    # count, which throws away rails, floor and people around the table.
    table_hsv_lower: tuple[int, int, int] | None = None
    table_hsv_upper: tuple[int, int, int] | None = None
    blur_kernel: int = 5  # Gaussian blur before masking; 0 or 1 disables
    morph_kernel: int = 5  # open/close to remove speckle and fill holes; 0 disables
    # Optional region of interest: polygon (pixel coords) covering the playing surface.
    roi_polygon: Sequence[tuple[int, int]] | None = None
    # Shrink the image by this factor before detecting (speed); results are scaled back up.
    downscale: float = 1.0

    @classmethod
    def from_json(cls, path: str | Path) -> "PuckDetectorConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, data: dict) -> "PuckDetectorConfig":
        data = dict(data)
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown detector config keys: {sorted(unknown)}")
        for key in ("hsv_lower", "hsv_upper", "table_hsv_lower", "table_hsv_upper"):
            if data.get(key) is not None:
                data[key] = tuple(data[key])
        if data.get("hsv_extra_range") is not None:
            lo, hi = data["hsv_extra_range"]
            data["hsv_extra_range"] = (tuple(lo), tuple(hi))
        if data.get("roi_polygon") is not None:
            data["roi_polygon"] = [tuple(p) for p in data["roi_polygon"]]
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


@dataclass(frozen=True)
class PuckDetection:
    x: float  # pixel column of the puck centre (full-resolution image)
    y: float  # pixel row of the puck centre
    radius: float  # pixels, from the min enclosing circle
    area: float  # pixels^2 of the blob
    circularity: float  # 4*pi*area / perimeter^2
    score: float  # higher is more puck-like


@dataclass
class _Candidate:
    detection: PuckDetection
    contour: np.ndarray = field(repr=False)


class PuckDetector:
    def __init__(self, config: PuckDetectorConfig | None = None):
        self.config = config or PuckDetectorConfig()
        self._roi_mask_cache: dict[tuple[int, int], np.ndarray] = {}
        self.last_table_hull: np.ndarray | None = None  # playing-surface outline from the last image

    # ---- public API -----------------------------------------------------------------------

    def detect(self, image: np.ndarray, previous: tuple[float, float] | None = None) -> PuckDetection | None:
        """Return the most puck-like blob in `image` (BGR), or None if nothing qualifies."""
        candidates = self.candidates(image)
        if not candidates:
            return None
        if previous is not None and len(candidates) > 1:
            px, py = previous
            return min(candidates, key=lambda c: (c.x - px) ** 2 + (c.y - py) ** 2)
        return candidates[0]

    def candidates(self, image: np.ndarray) -> list[PuckDetection]:
        """All blobs that pass the filters, best score first."""
        return [c.detection for c in self._find(image)]

    def mask(self, image: np.ndarray) -> np.ndarray:
        """Binary mask (uint8 0/255) of puck-coloured pixels, at full resolution. Useful for tuning."""
        small, _ = self._prepare(image)
        mask = self._mask(small)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    def draw(self, image: np.ndarray, detection: PuckDetection | None, color=(0, 255, 0)) -> np.ndarray:
        """Copy of `image` with the detection drawn on it."""
        out = image.copy()
        if detection is not None:
            centre = (int(round(detection.x)), int(round(detection.y)))
            cv2.circle(out, centre, max(int(round(detection.radius)), 2), color, 2)
            cv2.drawMarker(out, centre, color, cv2.MARKER_CROSS, 12, 2)
        if self.config.roi_polygon is not None:
            pts = np.array(self.config.roi_polygon, dtype=np.int32)
            cv2.polylines(out, [pts], True, (255, 200, 0), 1)
        if self.last_table_hull is not None:
            cv2.polylines(out, [self.last_table_hull], True, (0, 200, 255), 2)
        return out

    # ---- internals ------------------------------------------------------------------------

    def _find(self, image: np.ndarray) -> list[_Candidate]:
        cfg = self.config
        small, scale = self._prepare(image)
        mask = self._mask(small)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_scale = scale * scale
        out: list[_Candidate] = []
        for contour in contours:
            area = cv2.contourArea(contour) / area_scale
            if not cfg.min_area_px <= area <= cfg.max_area_px:
                continue
            perimeter = cv2.arcLength(contour, True) / scale
            if perimeter <= 0:
                continue
            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < cfg.min_circularity:
                continue
            if cfg.min_fill > 0 and _fill_ratio(mask, contour) < cfg.min_fill:
                continue
            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue
            x = m["m10"] / m["m00"] / scale
            y = m["m01"] / m["m00"] / scale
            _, r = cv2.minEnclosingCircle(contour)
            radius = r / scale
            score = circularity
            if cfg.expected_radius_px:
                # Penalise size mismatch: 1.0 at the expected radius, falling off either side.
                ratio = radius / cfg.expected_radius_px
                score *= float(np.exp(-((np.log(ratio)) ** 2) / 0.18))
            else:
                score *= float(np.log1p(area))  # among circular blobs, prefer bigger ones
            out.append(
                _Candidate(
                    PuckDetection(x=x, y=y, radius=radius, area=area, circularity=circularity, score=score),
                    contour,
                )
            )
        out.sort(key=lambda c: c.detection.score, reverse=True)
        return out

    def _prepare(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected a BGR image of shape (H, W, 3)")
        scale = 1.0
        if self.config.downscale and self.config.downscale > 1.0:
            scale = 1.0 / self.config.downscale
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        k = self.config.blur_kernel
        if k and k > 1:
            k = k | 1  # must be odd
            image = cv2.GaussianBlur(image, (k, k), 0)
        return image, scale

    def _mask(self, image: np.ndarray) -> np.ndarray:
        cfg = self.config
        if cfg.mode == "hsv":
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array(cfg.hsv_lower), np.array(cfg.hsv_upper))
            if cfg.hsv_extra_range is not None:
                lo, hi = cfg.hsv_extra_range
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        elif cfg.mode in ("bright", "dark"):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kind = cv2.THRESH_BINARY if cfg.mode == "bright" else cv2.THRESH_BINARY_INV
            _, mask = cv2.threshold(gray, cfg.threshold, 255, kind)
        else:
            raise ValueError(f"unknown mode {cfg.mode!r}")

        if cfg.morph_kernel and cfg.morph_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        d = cfg.downscale if cfg.downscale and cfg.downscale > 1.0 else 1.0
        if cfg.roi_polygon is not None:
            mask &= self._roi_mask(mask.shape[:2], 1.0 / d)
        if cfg.table_hsv_lower is not None and cfg.table_hsv_upper is not None:
            surface, hull = _table_surface(image, cfg.table_hsv_lower, cfg.table_hsv_upper)
            self.last_table_hull = None if hull is None else (hull.astype(np.float64) * d).astype(np.int32)
            mask &= surface
        return mask

    def _roi_mask(self, shape: tuple[int, int], scale: float) -> np.ndarray:
        key = shape
        if key not in self._roi_mask_cache:
            roi = np.zeros(shape, dtype=np.uint8)
            pts = (np.array(self.config.roi_polygon, dtype=np.float64) * scale).astype(np.int32)
            cv2.fillPoly(roi, [pts], 255)
            self._roi_mask_cache[key] = roi
        return self._roi_mask_cache[key]


def _fill_ratio(mask: np.ndarray, contour: np.ndarray) -> float:
    """Share of the area inside `contour` that is set in `mask` (1.0 = solid disk, low = ring)."""
    x, y, w, h = cv2.boundingRect(contour)
    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled, [contour - np.array([x, y])], -1, 255, thickness=cv2.FILLED)
    inside = cv2.countNonZero(filled)
    if inside == 0:
        return 0.0
    return cv2.countNonZero(cv2.bitwise_and(mask[y : y + h, x : x + w], filled)) / inside


def _table_surface(image: np.ndarray, lower, upper) -> tuple[np.ndarray, np.ndarray | None]:
    """Mask of the playing surface: convex hull of the largest region in the table colour.

    The hull (not the raw colour mask) is used so that a puck sitting on the table, which punches
    a hole in the table colour, still counts as inside.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colour = cv2.inRange(hsv, np.array(lower), np.array(upper))
    contours, _ = cv2.findContours(colour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    surface = np.zeros(image.shape[:2], dtype=np.uint8)
    if not contours:
        return surface, None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    cv2.fillConvexPoly(surface, hull, 255)
    return surface, hull
