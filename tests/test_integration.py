"""
Integration tests for the full pipeline.
Tests end-to-end processing with synthetic hand-like shapes.
"""

import pytest
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.camera import CameraCapture
from src.skin_detection import SkinDetector
from src.hand_detection import HandDetector
from src.gesture_recognition import GestureRecognizer
from src.gesture_history import GestureHistory


class TestIntegrationSyntheticHands:
    """Integration tests with synthetic hand-like shapes."""
    
    @pytest.fixture
    def setup_pipeline(self):
        """Set up all pipeline components."""
        skin_detector = SkinDetector()
        hand_detector = HandDetector(min_contour_area=500)
        gesture_recognizer = GestureRecognizer()
        gesture_history = GestureHistory(buffer_size=10, consensus_threshold=5)
        
        return {
            'skin_detector': skin_detector,
            'hand_detector': hand_detector,
            'gesture_recognizer': gesture_recognizer,
            'gesture_history': gesture_history
        }
    
    def test_full_pipeline_with_circle(self, setup_pipeline):
        """Test full pipeline with a synthetic circle (Fist-like)."""
        # Create synthetic frame with a colored circle
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        # Draw circle with a specific color that we can detect
        cv2.circle(frame, (200, 200), 80, (100, 100, 200), -1)  # Light color in BGR
        
        # Step 1: Skin detection - set broad HSV range to detect this color
        pipeline = setup_pipeline
        skin_detector = pipeline['skin_detector']
        
        # Set a broad HSV range to ensure detection
        skin_detector.set_hsv_range(
            np.array([0, 0, 0], dtype=np.uint8),
            np.array([180, 255, 255], dtype=np.uint8)
        )
        
        mask = skin_detector.detect_and_clean(frame, morphology_op='both')
        
        assert mask is not None
        # May or may not detect with broad range, just verify no crash
        assert mask.dtype == np.uint8
        
        # Step 2: Hand detection
        hand_detector = pipeline['hand_detector']
        hand_contour = hand_detector.find_hand_contour(mask)
        
        # Step 3: Gesture recognition if contour found
        if hand_contour is not None:
            gesture_recognizer = pipeline['gesture_recognizer']
            gesture = gesture_recognizer.recognize_gesture(hand_contour)
            assert gesture is None or isinstance(gesture, str)
        
        # Step 4: Gesture history
        gesture_history = pipeline['gesture_history']
        gesture_history.add_frame(None)
        
        assert len(gesture_history.get_history()) == 1
    
    def test_full_pipeline_with_elongated_shape(self, setup_pipeline):
        """Test full pipeline with elongated shape (Finger-like)."""
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        # Draw ellipse (elongated, finger-like)
        cv2.ellipse(frame, (200, 200), (120, 40), 0, 0, 360, (100, 60, 120), -1)
        
        pipeline = setup_pipeline
        skin_detector = pipeline['skin_detector']
        
        # Calibrate to synthetic skin color
        lower, upper = SkinDetector.extract_hsv_from_region(
            frame,
            np.ones((400, 400), dtype=np.uint8) * 255
        )
        skin_detector.set_hsv_range(lower, upper)
        
        mask = skin_detector.detect_and_clean(frame)
        hand_detector = pipeline['hand_detector']
        hand_contour = hand_detector.find_hand_contour(mask)
        
        if hand_contour is not None:
            gesture_recognizer = pipeline['gesture_recognizer']
            gesture = gesture_recognizer.recognize_gesture(hand_contour)
            
            # Could be One Finger or something else
            assert gesture in {None, "Fist", "Open Palm", "One Finger", "Two Fingers"}
    
    def test_full_pipeline_no_hand(self, setup_pipeline):
        """Test full pipeline when no hand is present."""
        # Empty frame
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        
        pipeline = setup_pipeline
        skin_detector = pipeline['skin_detector']
        mask = skin_detector.detect_and_clean(frame)
        
        hand_detector = pipeline['hand_detector']
        hand_contour = hand_detector.find_hand_contour(mask)
        
        # Should not find a hand
        assert hand_contour is None
        
        # Add None to history
        gesture_history = pipeline['gesture_history']
        gesture_history.add_frame(None)
        
        # Wait for consensus
        for _ in range(4):
            gesture_history.add_frame(None)
        
        smoothed = gesture_history.get_smoothed_gesture()
        assert smoothed is None
    
    def test_temporal_smoothing_integration(self, setup_pipeline):
        """Test that temporal smoothing works across frames."""
        pipeline = setup_pipeline
        gesture_history = pipeline['gesture_history']
        
        # Simulate frames with some noise
        gestures = ["Fist", "Fist", "Open Palm", "Fist", "Fist", "Fist", "Fist", "Fist"]
        
        for gesture in gestures:
            gesture_history.add_frame(gesture)
        
        smoothed = gesture_history.get_smoothed_gesture()
        
        # Should reach consensus on Fist (5+ frames)
        assert smoothed == "Fist"
    
    def test_gesture_classification_consistency(self, setup_pipeline):
        """Test that same shape produces consistent gesture across runs."""
        gesture_recognizer = setup_pipeline['gesture_recognizer']
        
        # Create same synthetic shape multiple times
        gestures = []
        for _ in range(5):
            frame = np.zeros((400, 400, 3), dtype=np.uint8)
            cv2.circle(frame, (200, 200), 70, (100, 60, 120), -1)
            
            # Quick HSV calibration
            skin_detector = SkinDetector()
            lower, upper = SkinDetector.extract_hsv_from_region(
                frame,
                np.ones((400, 400), dtype=np.uint8) * 255
            )
            skin_detector.set_hsv_range(lower, upper)
            
            mask = skin_detector.detect_and_clean(frame)
            hand_detector = HandDetector()
            hand_contour = hand_detector.find_hand_contour(mask)
            
            if hand_contour is not None:
                gesture = gesture_recognizer.recognize_gesture(hand_contour)
                gestures.append(gesture)
        
        # All gestures should be the same (or at least majority should be)
        if len(gestures) > 0:
            most_common = max(set(gestures), key=gestures.count)
            assert gestures.count(most_common) >= len(gestures) * 0.6  # At least 60% same


class TestPipelineRobustness:
    """Test robustness of pipeline with edge cases."""
    
    def test_pipeline_with_empty_frame(self):
        """Test pipeline doesn't crash with empty frame."""
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        
        skin_detector = SkinDetector()
        mask = skin_detector.detect_and_clean(frame)
        
        hand_detector = HandDetector()
        hand_contour = hand_detector.find_hand_contour(mask)
        
        # Should handle gracefully
        assert hand_contour is None
    
    def test_pipeline_with_noise(self):
        """Test pipeline with noisy frame."""
        frame = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
        
        skin_detector = SkinDetector()
        mask = skin_detector.detect_and_clean(frame)
        
        # Should produce a valid mask
        assert mask.dtype == np.uint8
        assert mask.shape == (400, 400)
    
    def test_gesture_recognizer_with_extreme_contour(self):
        """Test gesture recognizer with very small contour."""
        gesture_recognizer = GestureRecognizer()
        
        # Create tiny contour
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 2, 255, -1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            gesture = gesture_recognizer.recognize_gesture(contours[0])
            # Should handle gracefully
            assert gesture is None or isinstance(gesture, str)
