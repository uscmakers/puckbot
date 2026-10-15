"""Made-up test photos that look like our table: white puck and mallets on blue, grey floor."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from puckbot.vision.settings import DetectorSettings, ShapeRules, TableSettings

W, H = 640, 360  # photo size, pixels
TABLE_CORNERS = np.array([(100, 50), (540, 50), (540, 310), (100, 310)])  # TL, TR, BR, BL
PUCK_R, MALLET_R = 18, 30  # pixels

BLUE, WHITE, FLOOR = (200, 30, 10), (240, 240, 240), (90, 90, 90)  # BGR


def photo(pucks=(), mallets=(), extra=None) -> np.ndarray:
    """Floor + blue table + white pucks/mallets at the given (x, y) pixels."""
    image = np.full((H, W, 3), FLOOR, dtype=np.uint8)
    cv2.fillConvexPoly(image, TABLE_CORNERS, BLUE)
    for x, y in pucks:
        cv2.circle(image, (x, y), PUCK_R, WHITE, -1)
    for x, y in mallets:
        cv2.circle(image, (x, y), MALLET_R, WHITE, -1)
    if extra:
        extra(image)
    return image


@pytest.fixture
def detector_settings() -> DetectorSettings:
    return DetectorSettings(
        white_hsv_lower=(0, 0, 170),
        white_hsv_upper=(179, 90, 255),
        puck=ShapeRules(
            expected_radius_px=PUCK_R, min_area_px=300, max_area_px=1500, min_roundness=0.6, min_fill=0.85
        ),
        mallet=ShapeRules(
            expected_radius_px=MALLET_R, min_area_px=1800, max_area_px=5000, min_roundness=0.4, min_fill=0.5
        ),
    )


@pytest.fixture
def table_settings() -> TableSettings:
    return TableSettings(
        outer_width_in=40.0,
        outer_length_in=80.0,
        rail_width_in=0.0,
        surface_hsv_lower=(100, 120, 20),
        surface_hsv_upper=(130, 255, 255),
    )
