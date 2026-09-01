"""
Unit tests for skin detection module.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.skin_detection import SkinDetector


class TestSkinDetector:
    """Test suite for SkinDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a default detector instance."""
        return SkinDetector()

    def test_initialization(self, detector):
        """Test detector initialization and defaults."""
        assert detector.lower_hsv is not None
        assert detector.upper_hsv is not None
        assert len(detector.lower_hsv) == 3
        assert len(detector.upper_hsv) == 3

    def test_set_and_get_hsv_range(self, detector):
        """Test setting and retrieving custom HSV range."""
        lower = np.array([5, 20, 50], dtype=np.uint8)
        upper = np.array([15, 140, 255], dtype=np.uint8)

        detector.set_hsv_range(lower, upper)

        retrieved_lower, retrieved_upper = detector.get_hsv_range()
        assert np.array_equal(retrieved_lower, lower)
        assert np.array_equal(retrieved_upper, upper)

    def test_detect_skin_valid_frame(self, detector):
        """Test that detect_skin returns binary uint8 mask for valid frame."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = (255, 0, 0)  # Pure blue

        mask = detector.detect_skin(frame)

        assert mask is not None
        assert mask.dtype == np.uint8
        assert mask.shape == (100, 100)
        assert np.all((mask == 0) | (mask == 255))

    def test_invalid_and_empty_frame_handling(self, detector):
        """Test that invalid or empty frames raise ValueError."""
        with pytest.raises(ValueError, match="Invalid frame"):
            detector.detect_skin(None)

        with pytest.raises(ValueError, match="Invalid frame"):
            detector.detect_skin(np.empty((0, 0, 3), dtype=np.uint8))

    def test_invalid_morphology_operation_raises_value_error(self, detector):
        """Test that invalid morphology operation name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid morphology operation"):
            detector.apply_morphology(np.zeros((10, 10), dtype=np.uint8), "invalid_op")

    def test_hsv_segmentation(self):
        """Test normal HSV segmentation on skin-colored pixels."""
        detector = SkinDetector(
            lower_hsv=np.array([0, 15, 35], dtype=np.uint8),
            upper_hsv=np.array([30, 255, 255], dtype=np.uint8),
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        center_hsv = np.array([[[15, 100, 150]]], dtype=np.uint8)
        center_bgr = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2BGR)[0, 0]

        frame[25:75, 25:75] = center_bgr

        mask = detector.detect_skin(frame)
        assert np.sum(mask[25:75, 25:75] == 255) > 2000
        assert np.sum(mask[0:20, 0:20] == 255) == 0

    def test_wrap_around_segmentation(self):
        """Test hue wrap-around segmentation (e.g. lower=170, upper=10)."""
        detector = SkinDetector(
            lower_hsv=np.array([170, 100, 100], dtype=np.uint8),
            upper_hsv=np.array([10, 255, 255], dtype=np.uint8),
            blur_kernel_size=0,
        )
        # Test 3 pixels: Hue=175 (in wrap), Hue=5 (in wrap), Hue=40 (out of wrap)
        hsv = np.array([[[175, 200, 200], [5, 200, 200], [40, 200, 200]]], dtype=np.uint8)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        mask = detector.detect_skin(frame)
        assert mask[0].tolist() == [255, 255, 0]

    def test_apply_morphology_open_and_close(self, detector):
        """Test morphological open (noise removal) and close (hole filling)."""
        # Noise test for opening
        noise_mask = np.zeros((100, 100), dtype=np.uint8)
        noise_mask[10, 10] = 255
        noise_mask[20, 20] = 255
        opened = detector.apply_morphology(noise_mask, operation="open")
        assert np.sum(opened) == 0

        # Hole test for closing
        hole_mask = np.zeros((100, 100), dtype=np.uint8)
        hole_mask[20:80, 20:80] = 255
        hole_mask[45:55, 45:55] = 0
        closed = detector.apply_morphology(hole_mask, operation="close")
        assert np.sum(closed) >= np.sum(hole_mask)

    def test_extract_hsv_from_region(self):
        """Test extracting HSV range from a synthetic region mask."""
        frame_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv_color = np.array([[[15, 100, 150]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        frame_bgr[20:80, 20:80] = bgr_color

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        lower, upper = SkinDetector.extract_hsv_from_region(frame_bgr, mask)

        assert lower is not None
        assert upper is not None
        assert len(lower) == 3
        assert len(upper) == 3
        assert lower[1] >= 15
        assert lower[2] >= 35
