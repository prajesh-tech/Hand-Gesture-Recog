"""
Unit tests for hand detection module.
"""

import pytest
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.hand_detection import HandDetector


class TestHandDetector:
    """Test HandDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance."""
        return HandDetector(min_contour_area=500)
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.min_contour_area == 500
    
    def test_find_hand_contour_empty_mask(self, detector):
        """Test finding contour in empty mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        contour = detector.find_hand_contour(mask)
        
        assert contour is None
    
    def test_find_hand_contour_single_circle(self, detector):
        """Test finding contour for a single circle."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Draw a circle (hand-like shape)
        cv2.circle(mask, (100, 100), 50, 255, -1)
        
        contour = detector.find_hand_contour(mask)
        
        assert contour is not None
        assert len(contour) > 0
    
    def test_find_hand_contour_multiple_contours(self, detector):
        """Test that largest contour is selected."""
        mask = np.zeros((300, 300), dtype=np.uint8)
        # Draw small circle
        cv2.circle(mask, (50, 50), 20, 255, -1)
        # Draw large circle (should be selected)
        cv2.circle(mask, (200, 200), 60, 255, -1)
        
        contour = detector.find_hand_contour(mask)
        
        assert contour is not None
        area = cv2.contourArea(contour)
        # Should be the large circle (area ~pi*60^2 ≈ 11300)
        assert area > 10000

    def test_find_hand_contour_rejects_near_full_frame_background(self):
        detector = HandDetector(min_contour_area=100, min_contour_area_ratio=0.0, max_contour_area_ratio=0.80)
        mask = np.full((200, 200), 255, dtype=np.uint8)
        cv2.circle(mask, (100, 100), 25, 0, -1)
        assert detector.find_hand_contour(mask) is None

    def test_find_hand_contour_uses_resolution_scaled_minimum_area(self):
        detector = HandDetector(min_contour_area=10, min_contour_area_ratio=0.10)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 10, 255, -1)
        assert detector.find_hand_contour(mask) is None

    def test_analyze_mask_reports_contour_filtering_diagnostics(self):
        detector = HandDetector(min_contour_area=100, min_contour_area_ratio=0.0)
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 30, 255, -1)
        analysis = detector.analyze_mask(mask)
        assert analysis["contour_count"] == 1
        assert analysis["plausible_contour_count"] == 1
        assert analysis["selected_contour"] is not None
    
    def test_filter_contours(self, detector):
        """Test filtering contours by area."""
        mask = np.zeros((300, 300), dtype=np.uint8)
        # Draw small circle
        cv2.circle(mask, (50, 50), 10, 255, -1)
        # Draw medium circle
        cv2.circle(mask, (150, 150), 40, 255, -1)
        # Draw large circle
        cv2.circle(mask, (250, 250), 60, 255, -1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter with min area = 2000
        filtered = detector.filter_contours(contours, min_area=2000)
        
        # Should keep only medium and large circles
        assert len(filtered) <= len(contours)
    
    def test_get_bounding_box(self, detector):
        """Test bounding box extraction."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 40, 255, -1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        
        x, y, w, h = detector.get_bounding_box(contour)
        
        assert x >= 0
        assert y >= 0
        assert w > 0
        assert h > 0
    
    def test_get_contour_area(self, detector):
        """Test area calculation."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 50, 255, -1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        
        area = detector.get_contour_area(contour)
        
        # Circle with radius 50: area ≈ pi*50^2 ≈ 7854
        assert area > 7000
        assert area < 8500
    
    def test_get_contour_perimeter(self, detector):
        """Test perimeter calculation."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 50, 255, -1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        
        perimeter = detector.get_contour_perimeter(contour)
        
        # Circle with radius 50: perimeter ≈ 2*pi*50 ≈ 314
        assert perimeter > 250
        assert perimeter < 350
    
    def test_draw_contour(self, detector):
        """Test drawing contour on frame."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 40, 255, -1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        
        frame_drawn = detector.draw_contour(frame, contour, color=(0, 255, 0))
        
        # Frame should have changed (contour drawn)
        assert not np.array_equal(frame_drawn, frame)
    
    def test_draw_bounding_box(self, detector):
        """Test drawing bounding box."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 40, 255, -1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        
        frame_drawn = detector.draw_bounding_box(frame, contour, color=(255, 0, 0))
        
        # Frame should have changed
        assert not np.array_equal(frame_drawn, frame)
    
    def test_set_min_contour_area(self, detector):
        """Test updating min area threshold."""
        detector.set_min_contour_area(1000)
        assert detector.min_contour_area == 1000
        
        detector.set_min_contour_area(2000)
        assert detector.min_contour_area == 2000
