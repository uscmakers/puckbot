"""End-to-end check on the real photo in data/samples (skipped if the image is missing)."""

import json
from pathlib import Path

import cv2
import pytest

from puckbot.vision.analyze_image import analyze

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data/samples/sample_hockey_table_top_view.png"
CONFIG = json.loads((ROOT / "configs/our_table.json").read_text())


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not present")
def test_sample_photo_finds_table_puck_and_mallet():
    result = analyze(cv2.imread(str(SAMPLE)), CONFIG)
    assert result["calibration"] is not None
    # One puck just below the centre circle.
    u, v = result["puck"]["px"]
    assert abs(u - 555) < 6 and abs(v - 846) < 6
    # One mallet on the table, by the bottom goal. Mallets on the rails / in hands are ignored.
    assert len(result["mallets"]) == 1
    u, v = result["mallets"][0]["px"]
    assert abs(u - 292) < 8 and abs(v - 1420) < 8
    # Puck sits just past the middle of the 82.5 in long surface.
    x_in, y_in = result["puck"]["in"]
    assert 20 < x_in < 27 and 45 < y_in < 53
    assert result["problems"] == []
