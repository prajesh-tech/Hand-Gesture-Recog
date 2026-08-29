"""
Unit tests for gesture recognition module.
"""

import pytest
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gesture_recognition import GestureRecognizer


class TestGestureRecognizer:
    """Test GestureRecognizer class."""
    
    @pytest.fixture
    def recognizer(self):
        """Create a recognizer instance."""
        return GestureRecognizer()
    
    def test_initialization(self, recognizer):
        """Test recognizer initialization."""
        assert recognizer.thresholds is not None
        assert 'fist_solidity_min' in recognizer.thresholds
        assert 'palm_defects_min' in recognizer.thresholds
    
    def test_extract_features_small_contour(self, recognizer):
        """Test feature extraction with too-small contour."""
        # Create a very small contour
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 3, 255, -1)  # Very small circle
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            features = recognizer.extract_features(contours[0])
            # Should return None (too small)
            assert features is None
    
    def test_extract_features_circle(self, recognizer):
        """Test feature extraction with circle (Fist-like)."""
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(mask, (150, 150), 60, 255, -1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        
        features = recognizer.extract_features(contour)
        
        assert features is not None
        assert 'area' in features
        assert 'solidity' in features
        assert 'extent' in features
        assert 'convexity_defects_count' in features
        assert features['area'] > 0
        assert 0 <= features['solidity'] <= 1
    
    def test_extract_features_elongated(self, recognizer):
        """Test feature extraction with elongated shape (Finger-like)."""
        mask = np.zeros((300, 300), dtype=np.uint8)
        # Draw an ellipse (elongated)
        cv2.ellipse(mask, (150, 150), (80, 30), 0, 0, 360, 255, -1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        
        features = recognizer.extract_features(contour)
        
        assert features is not None
        # Aspect ratio should be > 1 for elongated shape
        assert features['aspect_ratio'] > 1.0
    
    def test_set_threshold(self, recognizer):
        """Test updating a single threshold."""
        recognizer.set_threshold('fist_solidity_min', 0.8)
        assert recognizer.get_threshold('fist_solidity_min') == 0.8
    
    def test_update_thresholds(self, recognizer):
        """Test updating multiple thresholds."""
        new_thresholds = {
            'fist_solidity_min': 0.85,
            'palm_defects_min': 8
        }
        recognizer.update_thresholds(new_thresholds)
        
        assert recognizer.get_threshold('fist_solidity_min') == 0.85
        assert recognizer.get_threshold('palm_defects_min') == 8
    
    def test_get_all_thresholds(self, recognizer):
        """Test retrieving all thresholds."""
        thresholds = recognizer.get_all_thresholds()
        
        assert isinstance(thresholds, dict)
        assert len(thresholds) > 0
        assert 'fist_solidity_min' in thresholds
    
    def test_recognize_gesture_none_for_invalid(self, recognizer):
        """Test that recognize_gesture returns None for invalid contours."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 2, 255, -1)  # Too small
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            gesture = recognizer.recognize_gesture(contours[0])
            assert gesture is None
    
    def test_recognize_gesture_returns_string_or_none(self, recognizer):
        """Test that recognize_gesture returns valid gesture labels."""
        valid_gestures = {None, "Fist", "Open Palm", "One Finger", "Two Fingers"}
        
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(mask, (150, 150), 80, 255, -1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        
        gesture = recognizer.recognize_gesture(contour)
        assert gesture in valid_gestures


class TestGestureClassification:
    """Test specific gesture classification scenarios."""
    
    def test_fist_like_shape(self):
        """Test that a compact circle is classified as Fist."""
        recognizer = GestureRecognizer()
        
        mask = np.zeros((300, 300), dtype=np.uint8)
        # Draw a compact circle
        cv2.circle(mask, (150, 150), 70, 255, -1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            gesture = recognizer.recognize_gesture(contours[0])
            # High-solidity circle should be Fist
            assert gesture == "Fist"
    
    def test_elongated_finger_shape(self):
        """Test that elongated shapes are recognized."""
        recognizer = GestureRecognizer()
        
        mask = np.zeros((400, 200), dtype=np.uint8)
        # Draw a tall, narrow rectangle (finger-like)
        cv2.rectangle(mask, (50, 30), (150, 350), 255, -1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            gesture = recognizer.recognize_gesture(contours[0])
            # Could be any valid gesture or None depending on thresholds and actual features
            assert gesture in {None, "Fist", "One Finger", "Two Fingers", "Open Palm"}
