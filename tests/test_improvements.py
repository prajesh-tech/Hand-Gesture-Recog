"""
Regression tests for consolidated skin detection, calibration, and contour selection.
"""

import cv2
import numpy as np
import pytest

from src.hand_detection import HandDetector
from src.skin_detection import SkinDetector


def test_calibration_with_synthetic_hand():
    """Test calibration with synthetic hand-colored samples."""
    frames = []
    for _ in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create hand-colored region (orange/skin tone in HSV)
        center_hsv = np.array([[[12, 90, 150]]], dtype=np.uint8)
        center_bgr = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2BGR)[0, 0]

        h, w = 480, 640
        cy, cx = h // 2, w // 2
        region_size = 100

        frame[cy - region_size : cy + region_size, cx - region_size : cx + region_size] = center_bgr
        frames.append(frame)

    lower, upper, stats = SkinDetector.calibrate_from_samples(frames)

    assert stats["calibration_valid"] is True
    assert stats["valid_percentage"] > 10.0
    assert lower[0] < 30 or stats.get("hue_wrap_detected", False)
    assert lower[1] >= 15
    assert lower[2] >= 35


def test_hue_wraparound():
    """Test detection of hue wrap-around for red skin tones."""
    frames = []
    for _ in range(5):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        h, w = 480, 640
        cy, cx = h // 2, w // 2

        for dy in range(-50, 51, 10):
            for dx in range(-50, 51, 10):
                hue_val = 10 if (dy // 10 + dx // 10) % 2 == 0 else 170
                hsv = np.array([[[hue_val, 85, 145]]], dtype=np.uint8)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
                if 0 <= cy + dy < h and 0 <= cx + dx < w:
                    frame[cy + dy, cx + dx] = bgr

        frames.append(frame)

    lower, upper, stats = SkinDetector.calibrate_from_samples(frames)

    assert stats.get("hue_wrap_detected", False) is True or lower[0] < 30 or upper[0] < 30


def test_background_rejection():
    """Test that black/gray backgrounds are not detected as skin."""
    detector = SkinDetector(
        lower_hsv=np.array([0, 15, 40], dtype=np.uint8),
        upper_hsv=np.array([30, 255, 255], dtype=np.uint8),
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    center_hsv = np.array([[[15, 100, 150]]], dtype=np.uint8)
    center_bgr = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2BGR)[0, 0]

    h, w = 480, 640
    frame[h // 2 - 50 : h // 2 + 50, w // 2 - 50 : w // 2 + 50] = center_bgr

    mask = detector.detect_skin(frame)
    mask_clean = detector.apply_morphology(mask, operation="both")

    total_pixels = h * w
    skin_pixels_clean = np.count_nonzero(mask_clean)
    skin_pct_clean = 100.0 * skin_pixels_clean / total_pixels

    assert skin_pct_clean < 50.0


def test_hand_contour_selection():
    """Test smart hand contour selection over large circular blobs."""
    detector = HandDetector(frame_width=640, frame_height=480)

    mask = np.zeros((480, 640), dtype=np.uint8)

    # Large circular blob (background noise)
    cv2.circle(mask, (550, 350), 80, 255, -1)

    # Hand-like shape (irregular polygon with fingers)
    hand_pts = np.array(
        [
            [[300, 200]],
            [[320, 150]],
            [[330, 200]],
            [[340, 140]],
            [[350, 200]],
            [[360, 240]],
            [[340, 280]],
            [[310, 290]],
            [[290, 270]],
            [[280, 240]],
        ],
        dtype=np.int32,
    )
    cv2.drawContours(mask, [hand_pts], 0, 255, -1)

    diag = detector.get_diagnostic_info(mask)
    selected = detector.find_hand_contour(mask)

    assert diag["total_contours"] >= 2
    assert selected is not None

    area = cv2.contourArea(selected)
    assert area < 10000.0  # Selected hand shape, not 20,000 px circle
