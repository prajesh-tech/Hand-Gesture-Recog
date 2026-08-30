"""
Unit tests for skin detection module.
"""

import pytest
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.skin_detection import SkinDetector


class TestSkinDetector:
    """Test SkinDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance."""
        return SkinDetector()
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.lower_hsv is not None
        assert detector.upper_hsv is not None
        assert len(detector.lower_hsv) == 3
        assert len(detector.upper_hsv) == 3
    
    def test_set_hsv_range(self, detector):
        """Test setting custom HSV range."""
        lower = np.array([5, 20, 50], dtype=np.uint8)
        upper = np.array([15, 140, 255], dtype=np.uint8)
        
        detector.set_hsv_range(lower, upper)
        
        retrieved_lower, retrieved_upper = detector.get_hsv_range()
        assert np.array_equal(retrieved_lower, lower)
        assert np.array_equal(retrieved_upper, upper)
    
    def test_detect_skin_returns_mask(self, detector):
        """Test that detect_skin returns a binary mask."""
        # Create a simple test frame (all blue)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = (255, 0, 0)  # BGR: blue
        
        mask = detector.detect_skin(frame)
        
        assert mask is not None
        assert mask.dtype == np.uint8
        assert mask.shape == (100, 100)
        assert np.all((mask == 0) | (mask == 255))  # Binary mask
    
    def test_apply_morphology_open(self, detector):
        """Test morphological opening (removes small noise)."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Add noise (small white spots)
        mask[10, 10] = 255
        mask[20, 20] = 255
        
        cleaned = detector.apply_morphology(mask, operation='open')
        
        # Noise should be removed
        assert np.sum(cleaned) == 0 or np.sum(cleaned) < np.sum(mask)
    
    def test_apply_morphology_close(self, detector):
        """Test morphological closing (fills holes)."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Add a square with a small hole
        mask[20:80, 20:80] = 255
        mask[45:55, 45:55] = 0  # Small hole in center
        
        closed = detector.apply_morphology(mask, operation='close')
        
        # After closing, more pixels should be white (hole filled)
        # May not fill 100% depending on kernel, but should improve
        assert np.sum(closed) >= np.sum(mask)
    
    def test_detect_and_clean(self, detector):
        """Test convenience method detect_and_clean."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = (255, 0, 0)  # Blue frame
        
        mask = detector.detect_and_clean(frame, morphology_op='both')
        
        assert mask is not None
        assert mask.dtype == np.uint8
        assert mask.shape == (100, 100)
    
    def test_extract_hsv_from_region(self):
        """Test extracting HSV range from a region."""
        # Create a test image with a specific color region
        frame_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_bgr[20:80, 20:80] = (120, 100, 150)  # Some BGR color
        
        # Mask for the colored region
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        
        lower, upper = SkinDetector.extract_hsv_from_region(frame_bgr, mask)
        
        assert lower is not None
        assert upper is not None
        assert len(lower) == 3
        assert len(upper) == 3
        # Lower should be <= upper for each channel
        assert np.all(lower <= upper)
    
    def test_extract_hsv_from_empty_region(self):
        """Test extracting HSV from empty region (no mask)."""
        frame_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)  # Empty mask
        
        lower, upper = SkinDetector.extract_hsv_from_region(frame_bgr, mask)
        
        # Should return default values
        assert lower is not None
        assert upper is not None

    def test_hue_wrap_range_detects_both_red_boundaries(self):
        detector = SkinDetector(
            np.array([170, 100, 100], dtype=np.uint8),
            np.array([10, 255, 255], dtype=np.uint8),
            blur_kernel_size=0,
        )
        hsv = np.array([[[175, 200, 200], [5, 200, 200], [40, 200, 200]]], dtype=np.uint8)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        assert detector.detect_skin(frame)[0].tolist() == [255, 255, 0]

    def test_invalid_frame_and_morphology_operation_raise_value_error(self, detector):
        with pytest.raises(ValueError):
            detector.detect_skin(np.empty((0, 0, 3), dtype=np.uint8))
        with pytest.raises(ValueError):
            detector.apply_morphology(np.zeros((10, 10), dtype=np.uint8), "invalid")

    def test_percentile_calibration_ignores_dark_background(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        frame[5:15, 5:15] = (80, 110, 180)
        lower, upper = SkinDetector.extract_hsv_from_region(frame, np.full((20, 20), 255, dtype=np.uint8))
        assert lower[1] >= 20
        assert lower[2] >= 30
        assert np.all(lower <= upper) or lower[0] > upper[0]

    def test_calibration_pixel_count_excludes_dark_and_desaturated_pixels(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[2:8, 2:8] = (80, 110, 180)
        count = SkinDetector.valid_calibration_pixel_count(frame, np.full((10, 10), 255, dtype=np.uint8))
        assert count == 36
