import cv2
import numpy as np
import pytest

from conftest import H, R, W, render
from puckbot.vision import PuckDetector, PuckDetectorConfig

RED = PuckDetectorConfig(mode="hsv", hsv_lower=(0, 120, 70), hsv_upper=(10, 255, 255))


@pytest.mark.parametrize("center", [(100, 100), (320.5, 180.5), (W - R - 1, H - R - 1)])
def test_finds_puck_centre(center):
    det = PuckDetector(RED).detect(render(center))
    assert det is not None
    assert abs(det.x - center[0]) < 1.5 and abs(det.y - center[1]) < 1.5
    assert abs(det.radius - R) < 2.5


def test_no_puck_returns_none():
    blank = np.full((H, W, 3), 235, dtype=np.uint8)
    assert PuckDetector(RED).detect(blank) is None


def test_ignores_non_round_blob():
    det = PuckDetector(RED).detect(render((150, 150), distractor=True))
    assert det is not None and abs(det.x - 150) < 1.5


def test_previous_hint_picks_nearest():
    image = render((100, 100))
    cv2.circle(image, (500, 250), R, (30, 30, 210), -1)  # second puck
    det = PuckDetector(RED).detect(image, previous=(490, 245))
    assert abs(det.x - 500) < 1.5


def test_downscale_keeps_full_res_coordinates():
    cfg = PuckDetectorConfig(**{**RED.__dict__, "downscale": 2.0})
    det = PuckDetector(cfg).detect(render((321, 181)))
    assert abs(det.x - 321) < 2.0 and abs(det.y - 181) < 2.0


def test_roi_excludes_outside():
    cfg = PuckDetectorConfig(**{**RED.__dict__, "roi_polygon": [(0, 0), (W // 2, 0), (W // 2, H), (0, H)]})
    assert PuckDetector(cfg).detect(render((500, 180))) is None
    assert PuckDetector(cfg).detect(render((100, 180))) is not None


def test_bright_mode():
    image = np.full((H, W, 3), 40, dtype=np.uint8)
    cv2.circle(image, (200, 150), R, (255, 255, 255), -1)
    det = PuckDetector(PuckDetectorConfig(mode="bright", threshold=200)).detect(image)
    assert abs(det.x - 200) < 1.5


def test_config_json_roundtrip(tmp_path):
    cfg = PuckDetectorConfig(mode="dark", threshold=60, roi_polygon=[(0, 0), (10, 0), (10, 10)])
    path = tmp_path / "puck.json"
    cfg.to_json(path)
    assert PuckDetectorConfig.from_json(path) == cfg


def test_min_fill_rejects_ring():
    image = np.full((H, W, 3), 235, dtype=np.uint8)
    cv2.circle(image, (200, 180), 30, (30, 30, 210), 6)  # red ring, like a mallet rim
    cv2.circle(image, (450, 180), R, (30, 30, 210), -1)  # solid puck
    det = PuckDetector(PuckDetectorConfig(**{**RED.__dict__, "min_fill": 0.8})).detect(image)
    assert abs(det.x - 450) < 1.5


def test_table_surface_ignores_off_table_blobs():
    image = np.full((H, W, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (100, 50), (540, 310), (200, 30, 10), -1)  # blue playing surface
    cv2.circle(image, (40, 40), R, (30, 30, 210), -1)  # red blob on the floor, bigger score
    cv2.circle(image, (300, 180), R - 4, (30, 30, 210), -1)  # puck on the table
    cfg = PuckDetectorConfig(**{**RED.__dict__, "table_hsv_lower": (100, 150, 60), "table_hsv_upper": (130, 255, 255)})
    detector = PuckDetector(cfg)
    det = detector.detect(image)
    assert abs(det.x - 300) < 1.5
    assert detector.last_table_hull is not None
