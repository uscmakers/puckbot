import cv2
import numpy as np

from puckbot.vision.calibration import TableCalibration, find_surface_corners

BLUE_LO, BLUE_HI = (100, 150, 60), (130, 255, 255)


def tilted_table(w=1000, h=1400):
    """A 400 x 800 mm blue surface seen at an angle (known homography)."""
    image = np.full((h, w, 3), 120, dtype=np.uint8)
    corners_px = np.array([[150, 60], [820, 130], [860, 1350], [180, 1300]], dtype=np.float32)
    cv2.fillConvexPoly(image, corners_px.astype(np.int32), (200, 30, 10))
    return image, corners_px


def test_surface_corners_found():
    image, truth = tilted_table()
    found = find_surface_corners(image, BLUE_LO, BLUE_HI)
    assert np.abs(found - truth).max() < 3.0


def test_px_to_mm_corners_and_round_trip():
    image, truth = tilted_table()
    cal = TableCalibration.from_surface(image, BLUE_LO, BLUE_HI, 400.0, 800.0)
    expected = [(0, 0), (400, 0), (400, 800), (0, 800)]
    for (u, v), (x, y) in zip(truth, expected):
        mx, my = cal.px_to_mm(u, v)
        assert abs(mx - x) < 5 and abs(my - y) < 5
    u, v = cal.mm_to_px(200, 300)
    assert np.allclose(cal.px_to_mm(u, v), (200, 300), atol=1e-6)


def test_json_round_trip(tmp_path):
    image, _ = tilted_table()
    cal = TableCalibration.from_surface(image, BLUE_LO, BLUE_HI, 400.0, 800.0)
    cal.to_json(tmp_path / "cal.json")
    back = TableCalibration.from_json(tmp_path / "cal.json")
    assert np.allclose(back.homography, cal.homography) and back.length_mm == 800.0
