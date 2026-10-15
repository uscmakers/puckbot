"""Puck and mallet detector, on made-up photos (see conftest.py)."""

import cv2
import pytest
from conftest import MALLET_R, PUCK_R, TABLE_CORNERS, WHITE, photo

from puckbot.vision import PuckMalletDetector


def find(settings, image, on_table=True):
    outline = TABLE_CORNERS if on_table else None
    return PuckMalletDetector(settings).find(image, table_outline=outline)


# ── Finds the right things ────────────────────────────────────────────────────


@pytest.mark.parametrize("at", [(150, 100), (320, 180), (500, 280)])
def test_finds_puck_centre_and_size(detector_settings, at):
    puck = find(detector_settings, photo(pucks=[at])).puck
    assert abs(puck.x - at[0]) < 1.5 and abs(puck.y - at[1]) < 1.5
    assert abs(puck.radius - PUCK_R) < 2.5


def test_tells_puck_and_mallet_apart(detector_settings):
    found = find(detector_settings, photo(pucks=[(200, 180)], mallets=[(420, 180)]))
    assert len(found.pucks) == 1 and abs(found.puck.x - 200) < 1.5
    assert len(found.mallets) == 1 and abs(found.mallets[0].x - 420) < 1.5
    assert abs(found.mallets[0].radius - MALLET_R) < 2.5


def test_empty_table(detector_settings):
    found = find(detector_settings, photo())
    assert found.puck is None and found.mallets == []


# ── Ignores the wrong things ──────────────────────────────────────────────────


def test_ignores_things_off_the_table(detector_settings):
    found = find(detector_settings, photo(pucks=[(40, 40), (300, 180)]))  # first one is on the floor
    assert len(found.pucks) == 1 and abs(found.puck.x - 300) < 1.5


def test_rejects_lines(detector_settings):
    def centre_line(image):
        cv2.rectangle(image, (110, 178), (530, 184), WHITE, -1)

    found = find(detector_settings, photo(extra=centre_line))
    assert found.puck is None
    assert found.shapes[0].kind == "rejected" and found.shapes[0].reason == "not round"


def test_rejects_rings(detector_settings):
    def ring(image):  # like the centre circle
        cv2.circle(image, (320, 180), MALLET_R, WHITE, 6)

    found = find(detector_settings, photo(extra=ring))
    assert found.mallets == []
    assert found.shapes[0].reason == "ring, not solid"
