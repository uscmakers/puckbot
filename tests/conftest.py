"""Synthetic test images: a red puck on a white table."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

W, H, R = 640, 360, 18
PUCK_BGR = (30, 30, 210)  # red


def render(center: tuple[float, float], distractor: bool = False) -> np.ndarray:
    image = np.full((H, W, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (W - 1, H - 1), (60, 60, 60), 6)  # rails
    cv2.line(image, (W // 2, 0), (W // 2, H), (200, 120, 120), 2)  # centre line (bluish)
    if distractor:
        cv2.rectangle(image, (500, 280), (540, 300), PUCK_BGR, -1)  # red but not round
    cv2.circle(image, (int(round(center[0])), int(round(center[1]))), R, PUCK_BGR, -1, lineType=cv2.LINE_AA)
    return image
