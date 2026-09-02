"""
Unit tests for gesture recognition module, heuristic confidence calculation, and result dataclasses.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gesture_recognition import GestureRecognizer
from src.results import GestureResult


class TestGestureRecognizer:
    """Test GestureRecognizer class."""

    @pytest.fixture
    def recognizer(self):
        """Create a recognizer instance."""
        return GestureRecognizer()

    def test_initialization(self, recognizer):
        """Test recognizer initialization."""
        assert recognizer.thresholds is not None
        assert "fist_solidity_min" in recognizer.thresholds
        assert "palm_defects_min" in recognizer.thresholds

    def test_extract_features_small_contour(self, recognizer):
        """Test feature extraction with too-small contour."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 3, 255, -1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            features = recognizer.extract_features(contours[0])
            assert features is None

    def test_extract_features_circle(self, recognizer):
        """Test feature extraction with circle (Fist-like)."""
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(mask, (150, 150), 60, 255, -1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]

        features = recognizer.extract_features(contour)

        assert features is not None
        assert "area" in features
        assert "solidity" in features
        assert "extent" in features
        assert "convexity_defects_count" in features
        assert features["area"] > 0
        assert 0 <= features["solidity"] <= 1

    def test_extract_features_elongated(self, recognizer):
        """Test feature extraction with elongated shape (Finger-like)."""
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.ellipse(mask, (150, 150), (80, 30), 0, 0, 360, 255, -1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]

        features = recognizer.extract_features(contour)

        assert features is not None
        assert features["aspect_ratio"] > 1.0

    def test_set_threshold(self, recognizer):
        """Test updating a single threshold."""
        recognizer.set_threshold("fist_solidity_min", 0.8)
        assert recognizer.get_threshold("fist_solidity_min") == 0.8

    def test_update_thresholds(self, recognizer):
        """Test updating multiple thresholds."""
        new_thresholds = {"fist_solidity_min": 0.85, "palm_defects_min": 8}
        recognizer.update_thresholds(new_thresholds)

        assert recognizer.get_threshold("fist_solidity_min") == 0.85
        assert recognizer.get_threshold("palm_defects_min") == 8

    def test_get_all_thresholds(self, recognizer):
        """Test retrieving all thresholds."""
        thresholds = recognizer.get_all_thresholds()

        assert isinstance(thresholds, dict)
        assert len(thresholds) > 0
        assert "fist_solidity_min" in thresholds

    def test_recognize_gesture_none_for_invalid(self, recognizer):
        """Test that recognize_gesture returns None for invalid contours."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 2, 255, -1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            gesture = recognizer.recognize_gesture(contours[0])
            assert gesture is None

    def test_calculate_confidence_range_and_clamping(self, recognizer):
        """Test confidence score calculation is clamped to 0.0 to 100.0%."""
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(mask, (150, 150), 60, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]

        features = recognizer.extract_features(contour)
        conf = recognizer.calculate_confidence(
            contour=contour,
            contour_score=4.0,
            features=features,
            gesture_label="Fist",
            history_confidence=1.0,
        )

        assert 0.0 <= conf <= 100.0
        assert conf > 50.0

        # None contour confidence must be 0.0
        none_conf = recognizer.calculate_confidence(None)
        assert none_conf == 0.0

    def test_process_gesture_returns_gesture_result(self, recognizer):
        """Test process_gesture returns GestureResult dataclass."""
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(mask, (150, 150), 60, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        res = recognizer.process_gesture(contours[0], contour_score=3.5, history_confidence=0.8)
        assert isinstance(res, GestureResult)
        assert res.is_hand_detected is True
        assert res.gesture == "Fist"
        assert 0.0 <= res.confidence <= 100.0


class TestGestureClassification:
    """Test specific gesture classification scenarios."""

    def test_fist_like_shape(self):
        """Test that a compact circle is classified as Fist."""
        recognizer = GestureRecognizer()

        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(mask, (150, 150), 70, 255, -1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            gesture = recognizer.recognize_gesture(contours[0])
            assert gesture == "Fist"

    def test_classifies_feature_combinations_and_unknown(self):
        recognizer = GestureRecognizer()
        base = {"area": 5000.0, "solidity": 0.9, "extent": 0.7, "elongation": 1.0}
        assert recognizer._classify_by_features(base | {"convexity_defects_count": 0}) == "Fist"
        assert recognizer._classify_by_features(base | {"convexity_defects_count": 3, "solidity": 0.7}) == "Open Palm"
        assert recognizer._classify_by_features(base | {"convexity_defects_count": 0, "elongation": 2.3}) == "One Finger"
        assert recognizer._classify_by_features(base | {"convexity_defects_count": 1, "elongation": 1.5}) == "Two Fingers"
        assert recognizer._classify_by_features(base | {"convexity_defects_count": 2, "elongation": 1.0}) == "Unknown"
