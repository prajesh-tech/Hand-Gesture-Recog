"""
Legacy classical HSV skin-color segmentation and contour analysis modules.
Preserved for comparative viva and college evaluation purposes.
"""

from src.legacy_hsv.skin_detection import SkinDetector
from src.legacy_hsv.hand_detection import HandDetector
from src.legacy_hsv.calibration import CalibrationManager

__all__ = ["SkinDetector", "HandDetector", "CalibrationManager"]
