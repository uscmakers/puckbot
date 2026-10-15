"""Vision: find the table, puck and mallets in a photo.

Files, in pipeline order:
    settings.py              what the table, puck and mallets look like
    table_calibration.py     find the table; convert pixels <-> inches
    puck_mallet_detector.py  find the puck and mallets
    pipeline.py              run the steps in order -> Analysis
    drawing.py               pictures of the result and of each step
    cli.py                   the `puckbot-analyze` command

Quick start:
    from puckbot.vision import analyze, load_settings
    result = analyze(cv2.imread("photo.png"), load_settings())
"""

from puckbot.vision.pipeline import Analysis, Located, analyze
from puckbot.vision.puck_mallet_detector import Detections, PuckMalletDetector, Shape
from puckbot.vision.settings import Settings, load_settings
from puckbot.vision.table_calibration import TableCalibration

__all__ = [
    "Analysis",
    "Detections",
    "Located",
    "PuckMalletDetector",
    "Settings",
    "Shape",
    "TableCalibration",
    "analyze",
    "load_settings",
]
