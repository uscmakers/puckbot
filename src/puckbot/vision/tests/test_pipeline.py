"""Whole pipeline on the real photo in data/samples/, plus the settings file."""

from pathlib import Path

import cv2
import pytest

from puckbot.vision import analyze, load_settings

SAMPLE = Path(__file__).resolve().parents[1] / "data/samples/sample_hockey_table_top_view.png"


def test_sample_photo():
    a = analyze(cv2.imread(str(SAMPLE)), load_settings())

    assert a.table is not None

    # One puck, just below the centre circle
    assert abs(a.puck.pixel[0] - 555) < 6 and abs(a.puck.pixel[1] - 846) < 6
    x, y = a.puck.position_in
    assert 20 < x < 27 and 45 < y < 53

    # One mallet, by the bottom goal. Mallets on the rails / in hands don't count.
    assert len(a.mallets) == 1
    assert abs(a.mallets[0].pixel[0] - 292) < 8 and abs(a.mallets[0].pixel[1] - 1420) < 8

    assert a.problems == []


def test_settings_typo_is_an_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"table": {"outer_widht_in": 50}}')
    with pytest.raises(ValueError, match="outer_widht_in"):
        load_settings(bad)
