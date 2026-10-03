"""
Unit tests for HandDetector contour filtering, min_contour_score rejection, and public input validation.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.legacy_hsv.hand_detection import HandDetector
from src.results import HandDetectionResult


class TestHandDetector:
    """Test suite for HandDetector class."""

    @pytest.fixture
    def detector(self):
        """Create standard HandDetector instance."""
        return HandDetector(min_contour_area=500.0, frame_width=640, frame_height=480)

    def test_find_valid_hand_contour(self, detector):
        """Test detection of a valid hand-shaped contour."""
        mask = np.zeros((480, 640), dtype=np.uint8)
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

        selected = detector.find_hand_contour(mask)
        assert selected is not None
        assert cv2.contourArea(selected) >= 500.0

    def test_minimum_area_rejection(self, detector):
        """Test rejection of contours smaller than min_contour_area."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        # Small square of 10x10 = 100 px (< 500 px min area)
        cv2.rectangle(mask, (100, 100), (110, 110), 255, -1)

        selected = detector.find_hand_contour(mask)
        assert selected is None

    def test_full_frame_rejection(self, detector):
        """Test rejection of near full-frame contours (background noise)."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(mask, (5, 5), (635, 475), 255, -1)

        selected = detector.find_hand_contour(mask)
        assert selected is None

    def test_strip_rejection(self, detector):
        """Test rejection of extreme aspect ratio strip contours along frame edge."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        # Vertical strip along edge (height=460, width=5)
        cv2.rectangle(mask, (0, 10), (5, 470), 255, -1)

        selected = detector.find_hand_contour(mask)
        assert selected is None

    def test_background_circular_blob_rejection(self, detector):
        """Test scoring penalty prefers hand shape over circular background blob."""
        mask = np.zeros((480, 640), dtype=np.uint8)

        # Large circular blob (circularity > 0.92)
        cv2.circle(mask, (500, 300), 70, 255, -1)

        # Hand-like irregular contour with fingers
        hand_pts = np.array(
            [
                [[200, 200]],
                [[220, 140]],
                [[230, 190]],
                [[240, 130]],
                [[250, 190]],
                [[260, 230]],
                [[240, 270]],
                [[210, 280]],
                [[190, 260]],
            ],
            dtype=np.int32,
        )
        cv2.drawContours(mask, [hand_pts], 0, 255, -1)

        selected = detector.find_hand_contour(mask)
        assert selected is not None
        assert cv2.contourArea(selected) < 10000.0

    def test_min_contour_score_rejection(self):
        """Test that candidate with score below min_contour_score is rejected."""
        detector = HandDetector(min_contour_score=10.0)  # Unattainable score threshold
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(mask, (300, 200), 40, 255, -1)

        selected = detector.find_hand_contour(mask)
        assert selected is None

    def test_invalid_mask_inputs(self, detector):
        """Test public input validation for invalid or empty mask objects."""
        with pytest.raises(TypeError):
            detector.find_hand_contour(None)

        with pytest.raises(ValueError):
            detector.find_hand_contour(np.empty((0, 0), dtype=np.uint8))

        with pytest.raises(ValueError):
            detector.find_hand_contour(np.zeros((100, 100, 3), dtype=np.uint8))  # 3D array

    def test_process_mask_returns_hand_detection_result(self, detector):
        """Test process_mask returns HandDetectionResult dataclass."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        res = detector.process_mask(mask)
        assert isinstance(res, HandDetectionResult)
        assert res.is_hand_detected is False
        assert res.selected_contour is None

    def test_get_bounding_box_and_draw_helpers(self, detector):
        """Test bounding box extraction and drawing utility methods."""
        contour = np.array([[[10, 20]], [[50, 20]], [[50, 80]], [[10, 80]]], dtype=np.int32)
        x, y, w, h = detector.get_bounding_box(contour)
        assert (x, y, w, h) == (10, 20, 41, 61)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        drawn = detector.draw_both(frame, contour)
        assert drawn is not None
        assert drawn.shape == (100, 100, 3)
