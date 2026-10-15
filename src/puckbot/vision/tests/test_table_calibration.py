"""Table calibration, on a made-up table photographed at an angle."""

import cv2
import numpy as np
from conftest import BLUE

from puckbot.vision.table_calibration import TableCalibration, find_surface_corners

CORNERS = np.array([[150, 60], [820, 130], [860, 1350], [180, 1300]], dtype=np.float64)  # TL, TR, BR, BL


def tilted_table() -> np.ndarray:
    image = np.full((1400, 1000, 3), 120, dtype=np.uint8)
    cv2.fillConvexPoly(image, CORNERS.astype(np.int32), BLUE)
    return image


def test_finds_corners(table_settings):
    found = find_surface_corners(tilted_table(), table_settings)
    assert np.abs(found - CORNERS).max() < 3.0


def test_corners_map_to_table_corners(table_settings):
    cal = TableCalibration.from_photo(tilted_table(), table_settings)  # 40 x 80 in surface
    for (u, v), real in zip(CORNERS, [(0, 0), (40, 0), (40, 80), (0, 80)], strict=True):
        assert np.allclose(cal.to_table(u, v), real, atol=0.2)


def test_round_trip(table_settings):
    cal = TableCalibration.from_photo(tilted_table(), table_settings)
    u, v = cal.to_pixel(20, 30)
    assert np.allclose(cal.to_table(u, v), (20, 30), atol=1e-6)
