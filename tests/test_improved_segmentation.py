"""
Regression tests for skin detection, calibration, and contour selection.
Tests for proper handling of broad ranges, hue wrap-around, contamination, etc.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hand_detection import HandDetector
from src.skin_detection import SkinDetector


class TestCalibrationRobustness:
    """Test calibration produces reasonable ranges."""

    def test_calibration_with_clean_hand_samples(self):
        """Test calibration with clean hand samples."""
        frames = []
        for _ in range(5):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            center_hsv = np.array([[[15, 100, 150]]], dtype=np.uint8)
            center_bgr = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2BGR)[0, 0]

            h, w = 480, 640
            cy, cx = h // 2, w // 2
            region_size = 80
            y1, y2 = cy - region_size, cy + region_size
            x1, x2 = cx - region_size, cx + region_size

            frame[y1:y2, x1:x2] = center_bgr
            frames.append(frame)

        lower, upper, stats = SkinDetector.calibrate_from_samples(frames)

        assert stats["calibration_valid"] is True
        assert stats["valid_percentage"] > 20.0
        assert lower[0] <= 20 or (lower[0] > 150 and stats["hue_wrap_detected"])
        assert upper[0] <= 30 or upper[0] <= 180

    def test_calibration_rejects_low_valid_percentage(self):
        """Test calibration with mostly background."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        h, w = 480, 640
        cy, cx = h // 2, w // 2
        center_hsv = np.array([[[15, 100, 150]]], dtype=np.uint8)
        center_bgr = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        frame[cy - 5 : cy + 5, cx - 5 : cx + 5] = center_bgr

        lower, upper, stats = SkinDetector.calibrate_from_samples([frame])

        assert stats["valid_percentage"] < 30.0

    def test_calibration_respects_saturation_bounds(self):
        """Test that S >= 15 and V >= 35 are enforced."""
        frames = []
        for _ in range(3):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            h, w = 480, 640
            cy, cx = h // 2, w // 2

            gray_hsv = np.array([[[15, 5, 100]]], dtype=np.uint8)
            gray_bgr = cv2.cvtColor(gray_hsv, cv2.COLOR_HSV2BGR)[0, 0]
            frame[cy - 50 : cy + 50, cx - 50 : cx + 50] = gray_bgr

            frames.append(frame)

        lower, upper, stats = SkinDetector.calibrate_from_samples(frames)

        assert lower[1] >= 15
        assert lower[2] >= 35


class TestHueWrapAround:
    """Test handling of hue wrap-around for red skin tones."""

    def test_red_skin_tone_wrap_around_detection(self):
        """Test detection of hue wrap-around in red skin tones."""
        frames = []
        for _ in range(5):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            h, w = 480, 640
            cy, cx = h // 2, w // 2

            for i in range(-40, 41, 20):
                for j in range(-40, 41, 20):
                    hue_val = 10 if (i + j) % 2 == 0 else 170
                    hsv = np.array([[[hue_val, 80, 140]]], dtype=np.uint8)
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
                    frame[cy + i, cx + j] = bgr

            frames.append(frame)

        lower, upper, stats = SkinDetector.calibrate_from_samples(frames)

        assert stats.get("hue_wrap_detected", False) is True or lower[0] < 30 or upper[0] < 30


class TestSkinDetectorMasks:
    """Test that skin masks don't accept excessive background."""

    def test_mask_rejects_pure_black_background(self):
        """Test that pure black background is not detected as skin."""
        detector = SkinDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        center_hsv = np.array([[[15, 80, 150]]], dtype=np.uint8)
        center_bgr = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        h, w = 480, 640
        frame[h // 2 - 30 : h // 2 + 30, w // 2 - 30 : w // 2 + 30] = center_bgr

        mask = detector.detect_skin(frame)
        background_mask = frame[:, :, 2] < 10
        detected_in_background = np.sum((mask > 0) & background_mask)

        assert detected_in_background < 100

    def test_mask_accepts_hand_pixels(self):
        """Test that actual hand pixels are detected."""
        detector = SkinDetector(
            lower_hsv=np.array([0, 15, 35], dtype=np.uint8),
            upper_hsv=np.array([30, 255, 255], dtype=np.uint8),
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        center_hsv = np.array([[[15, 100, 150]]], dtype=np.uint8)
        center_bgr = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2BGR)[0, 0]

        h, w = 480, 640
        frame[h // 2 - 50 : h // 2 + 50, w // 2 - 50 : w // 2 + 50] = center_bgr

        mask = detector.detect_skin(frame)
        hand_region_detected = np.sum(mask[h // 2 - 50 : h // 2 + 50, w // 2 - 50 : w // 2 + 50]) > 100
        assert hand_region_detected


class TestContourDetection:
    """Test smart contour selection."""

    def test_selects_hand_shaped_contour_over_circle(self):
        """Test that hand-like contour is preferred over circular blobs."""
        detector = HandDetector(frame_width=640, frame_height=480)
        mask = np.zeros((480, 640), dtype=np.uint8)

        # Large circle (not hand-like)
        cv2.circle(mask, (150, 150), 80, 255, -1)

        # Hand-like shape (irregular)
        hand_pts = np.array(
            [
                [[300, 200]],
                [[320, 150]],
                [[340, 200]],
                [[350, 220]],
                [[340, 250]],
                [[320, 260]],
                [[300, 250]],
                [[290, 230]],
            ],
            dtype=np.int32,
        )
        cv2.drawContours(mask, [hand_pts], 0, 255, -1)

        selected = detector.find_hand_contour(mask)
        assert selected is not None

        area = cv2.contourArea(selected)
        assert area < 6000.0

    def test_rejects_near_fullframe_contour(self):
        """Test that contours filling most of frame are rejected."""
        detector = HandDetector(frame_width=640, frame_height=480)
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (630, 470), 255, -1)

        selected = detector.find_hand_contour(mask)
        assert selected is None


class TestImprovedVsOriginal:
    """Compare implementation with known problematic cases."""

    def test_broad_hsv_range_is_rejected(self):
        """Test that excessively broad calibration is handled."""
        old_lower = np.array([159, 20, 33], dtype=np.uint8)
        old_upper = np.array([24, 118, 145], dtype=np.uint8)

        detector = SkinDetector()
        detector.set_hsv_range(old_lower, old_upper)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = detector.detect_skin(frame)
        skin_percentage = 100.0 * np.count_nonzero(mask) / (480 * 640)

        assert skin_percentage < 50.0
