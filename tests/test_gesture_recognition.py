"""
Unit tests for explainable 3D LandmarkGestureRecognizer.
Tests spatial invariance (rotation, translation, scaling), boundary cases,
decoupled execution without MediaPipe dependencies, and heuristic confidence scoring.
"""

import math
import os
import sys
from typing import Dict, List
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gesture_recognition import GestureRecognizer, LandmarkGestureRecognizer
from src.results import GestureResult


def create_mock_hand(
    thumb_ext: bool = True,
    index_ext: bool = True,
    middle_ext: bool = True,
    ring_ext: bool = True,
    pinky_ext: bool = True,
) -> List[Dict[str, float]]:
    """Helper to generate a canonical 21-landmark hand configuration."""
    lms: List[Dict[str, float]] = []

    # 0: Wrist
    lms.append({"id": 0, "x": 0.50, "y": 0.80, "z": 0.0})

    # Thumb 1..4
    lms.append({"id": 1, "x": 0.45, "y": 0.75, "z": 0.0})
    lms.append({"id": 2, "x": 0.42, "y": 0.70, "z": 0.0})
    lms.append({"id": 3, "x": 0.38, "y": 0.65, "z": 0.0})
    if thumb_ext:
        lms.append({"id": 4, "x": 0.34, "y": 0.60, "z": 0.0})  # spread outward
    else:
        lms.append({"id": 4, "x": 0.50, "y": 0.68, "z": 0.0})  # tucked into palm

    # Index 5..8
    lms.append({"id": 5, "x": 0.45, "y": 0.60, "z": 0.0})
    lms.append({"id": 6, "x": 0.44, "y": 0.50, "z": 0.0})
    lms.append({"id": 7, "x": 0.43, "y": 0.40, "z": 0.0})
    lms.append({
        "id": 8,
        "x": 0.42 if index_ext else 0.45,
        "y": 0.30 if index_ext else 0.62,
        "z": 0.0,
    })

    # Middle 9..12
    lms.append({"id": 9, "x": 0.50, "y": 0.58, "z": 0.0})
    lms.append({"id": 10, "x": 0.50, "y": 0.48, "z": 0.0})
    lms.append({"id": 11, "x": 0.50, "y": 0.38, "z": 0.0})
    lms.append({
        "id": 12,
        "x": 0.50,
        "y": 0.28 if middle_ext else 0.60,
        "z": 0.0,
    })

    # Ring 13..16
    lms.append({"id": 13, "x": 0.55, "y": 0.60, "z": 0.0})
    lms.append({"id": 14, "x": 0.56, "y": 0.50, "z": 0.0})
    lms.append({"id": 15, "x": 0.57, "y": 0.40, "z": 0.0})
    lms.append({
        "id": 16,
        "x": 0.58 if ring_ext else 0.55,
        "y": 0.30 if ring_ext else 0.62,
        "z": 0.0,
    })

    # Pinky 17..20
    lms.append({"id": 17, "x": 0.60, "y": 0.63, "z": 0.0})
    lms.append({"id": 18, "x": 0.62, "y": 0.55, "z": 0.0})
    lms.append({"id": 19, "x": 0.63, "y": 0.47, "z": 0.0})
    lms.append({
        "id": 20,
        "x": 0.64 if pinky_ext else 0.60,
        "y": 0.39 if pinky_ext else 0.65,
        "z": 0.0,
    })

    return lms


def apply_rigid_transformation(
    landmarks: List[Dict[str, float]],
    rotation_matrix: np.ndarray,
    scale: float = 1.0,
    translation: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> List[Dict[str, float]]:
    """Apply consistent rigid-body 3D transformation (s * R * P + T) to all landmarks."""
    transformed: List[Dict[str, float]] = []
    for lm in landmarks:
        vec = np.array([lm["x"], lm["y"], lm.get("z", 0.0)], dtype=np.float64)
        new_vec = scale * (rotation_matrix @ vec) + translation
        transformed.append({
            "id": lm["id"],
            "x": float(new_vec[0]),
            "y": float(new_vec[1]),
            "z": float(new_vec[2]),
        })
    return transformed


def rotation_matrix_z(theta_deg: float) -> np.ndarray:
    """Create 3x3 rotation matrix around Z-axis."""
    rad = math.radians(theta_deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def rotation_matrix_arbitrary_axis(axis: np.ndarray, theta_deg: float) -> np.ndarray:
    """Create 3x3 rotation matrix around an arbitrary 3D axis (Rodrigues' rotation)."""
    rad = math.radians(theta_deg)
    u = axis / np.linalg.norm(axis)
    ux, uy, uz = u[0], u[1], u[2]
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c + ux*ux*(1-c), ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c), uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)],
    ], dtype=np.float64)


class TestGestureRecognition:
    """Test suite for LandmarkGestureRecognizer."""

    @pytest.fixture
    def recognizer(self) -> LandmarkGestureRecognizer:
        return LandmarkGestureRecognizer()

    def test_decoupling_no_mediapipe_imported(self):
        """Verify LandmarkGestureRecognizer operates without importing mediapipe."""
        # Check module dependencies
        import inspect
        src_code = inspect.getsource(LandmarkGestureRecognizer)
        assert "import mediapipe" not in src_code
        assert "from mediapipe" not in src_code

    def test_backward_compatibility_alias(self):
        """Test GestureRecognizer alias points to LandmarkGestureRecognizer."""
        assert GestureRecognizer is LandmarkGestureRecognizer

    def test_canonical_gesture_classifications(self, recognizer):
        """Verify all 4 canonical gestures and edge states classify correctly."""
        # Open Palm: all fingers extended & thumb extended
        open_palm = create_mock_hand(True, True, True, True, True)
        assert recognizer.recognize_gesture(open_palm) == "Open Palm"

        # Fist: all fingers closed & thumb tucked
        fist = create_mock_hand(False, False, False, False, False)
        assert recognizer.recognize_gesture(fist) == "Fist"

        # One Finger: Index extended; Middle, Ring, Pinky closed
        one_finger = create_mock_hand(False, True, False, False, False)
        assert recognizer.recognize_gesture(one_finger) == "One Finger"

        # Two Fingers: Index and Middle extended; Ring, Pinky closed
        two_fingers = create_mock_hand(False, True, True, False, False)
        assert recognizer.recognize_gesture(two_fingers) == "Two Fingers"

        # Unknown Gesture: e.g., only Pinky extended
        pinky_only = create_mock_hand(False, False, False, False, True)
        assert recognizer.recognize_gesture(pinky_only) == "Unknown"

        # No hand present
        assert recognizer.recognize_gesture(None) is None
        assert recognizer.recognize_gesture([]) is None

    def test_spatial_invariance_rotations(self, recognizer):
        """
        Verify gesture classifications are strictly invariant under 2D and 3D rotations.
        Applies rotation matrices consistently to the entire landmark set.
        """
        gestures = [
            ("Open Palm", create_mock_hand(True, True, True, True, True)),
            ("Fist", create_mock_hand(False, False, False, False, False)),
            ("One Finger", create_mock_hand(False, True, False, False, False)),
            ("Two Fingers", create_mock_hand(False, True, True, False, False)),
        ]

        # Test varying angles in plane (tilt / upside-down)
        test_angles = [30.0, 45.0, 90.0, 135.0, 180.0, 270.0, -60.0]
        for expected_label, hand in gestures:
            for angle in test_angles:
                rot = rotation_matrix_z(angle)
                rotated_hand = apply_rigid_transformation(hand, rot)
                assert (
                    recognizer.recognize_gesture(rotated_hand) == expected_label
                ), f"Failed for {expected_label} at {angle} deg rotation"

        # Test arbitrary 3D rotation out-of-plane
        axis_3d = np.array([1.0, 1.0, 0.5])
        rot_3d = rotation_matrix_arbitrary_axis(axis_3d, 35.0)
        for expected_label, hand in gestures:
            rotated_3d = apply_rigid_transformation(hand, rot_3d)
            assert (
                recognizer.recognize_gesture(rotated_3d) == expected_label
            ), f"Failed for {expected_label} with 3D rotation"

    def test_spatial_invariance_translation_and_scale(self, recognizer):
        """
        Verify gesture classifications are invariant under spatial translation and uniform scale.
        """
        gestures = [
            ("Open Palm", create_mock_hand(True, True, True, True, True)),
            ("Fist", create_mock_hand(False, False, False, False, False)),
            ("One Finger", create_mock_hand(False, True, False, False, False)),
            ("Two Fingers", create_mock_hand(False, True, True, False, False)),
        ]

        scales = [0.4, 0.75, 1.5, 2.5]
        translations = [
            np.array([0.2, -0.3, 0.1]),
            np.array([-0.5, 0.4, -0.2]),
            np.array([10.0, 25.0, -5.0]),
        ]

        identity = np.eye(3, dtype=np.float64)

        for expected_label, hand in gestures:
            for s in scales:
                for t in translations:
                    transformed = apply_rigid_transformation(hand, identity, scale=s, translation=t)
                    assert (
                        recognizer.recognize_gesture(transformed) == expected_label
                    ), f"Failed for {expected_label} with scale {s} and translation {t}"

    def test_boundary_conditions_extension_ratio(self, recognizer):
        """Test borderline finger extension ratios."""
        # Index finger: dist(Wrist, PIP) = 0.3059
        # Tip just below threshold (1.34) vs just above threshold (1.36)
        hand = create_mock_hand(False, False, False, False, False)
        wrist = hand[0]
        pip = hand[6]
        dist_wrist_pip = recognizer._dist_3d(wrist, pip)

        # 1. Just below threshold (1.34 * dist) -> closed
        sub_thresh_dist = dist_wrist_pip * 1.34
        hand[8] = {"id": 8, "x": wrist["x"], "y": wrist["y"] - sub_thresh_dist, "z": 0.0}
        eval_closed = recognizer.evaluate_finger(
            recognizer._get_landmark_map(hand), 5, 6, 8, "index"
        )
        assert eval_closed["is_extended"] is False

        # 2. Just above threshold (1.36 * dist) -> extended (aligned vertically)
        super_thresh_dist = dist_wrist_pip * 1.36
        hand[8] = {"id": 8, "x": pip["x"], "y": wrist["y"] - super_thresh_dist, "z": 0.0}
        eval_open = recognizer.evaluate_finger(
            recognizer._get_landmark_map(hand), 5, 6, 8, "index"
        )
        assert eval_open["is_extended"] is True

    def test_curved_bent_finger_rejected_by_straightness(self, recognizer):
        """
        Verify that a finger with adequate Euclidean distance but bent/curved joint alignment
        (cos theta <= 0.85) is correctly evaluated as NOT extended.
        """
        hand = create_mock_hand(False, True, False, False, False)
        # Turn the index finger at a sharp 90-degree right angle at PIP (6->7->8)
        # Vector 5->6 is (0, -0.10, 0), Vector 6->8 is (0.15, 0, 0)
        hand[8] = {"id": 8, "x": hand[6]["x"] + 0.15, "y": hand[6]["y"], "z": 0.0}

        lm_map = recognizer._get_landmark_map(hand)
        eval_bent = recognizer.evaluate_finger(lm_map, 5, 6, 8, "index")

        # Even if ratio > 1.35, straightness cos is ~0.0 < 0.85
        assert eval_bent["straightness"] < 0.85
        assert eval_bent["is_extended"] is False

    def test_thumb_spread_boundary(self, recognizer):
        """Test borderline thumb spread ratio."""
        hand = create_mock_hand(False, False, False, False, False)
        wrist = hand[0]
        pinky_mcp = hand[17]
        dist_w_p = recognizer._dist_3d(wrist, pinky_mcp)

        # Borderline tucked (0.84)
        hand[4] = {"id": 4, "x": pinky_mcp["x"] - dist_w_p * 0.84, "y": pinky_mcp["y"], "z": 0.0}
        tucked_eval = recognizer.evaluate_thumb(recognizer._get_landmark_map(hand))
        assert tucked_eval["is_extended"] is False

        # Borderline extended (0.86)
        hand[4] = {"id": 4, "x": pinky_mcp["x"] - dist_w_p * 0.86, "y": pinky_mcp["y"], "z": 0.0}
        ext_eval = recognizer.evaluate_thumb(recognizer._get_landmark_map(hand))
        assert ext_eval["is_extended"] is True

    def test_confidence_calculation_and_clamping(self, recognizer):
        """Test heuristic confidence evaluation, sub-score breakdown, and clamping to 0-100%."""
        # 1. None hand
        conf, bd = recognizer.calculate_confidence(None, None, None)
        assert conf == 0.0
        assert bd["quality"] == 0.0
        assert bd["rule_match"] == 0.0
        assert bd["stability"] == 0.0

        # 2. Open Palm with full consensus
        hand = create_mock_hand(True, True, True, True, True)
        features = recognizer.extract_features(hand)
        conf_full, bd_full = recognizer.calculate_confidence(
            hand, features, "Open Palm", history_confidence=1.0
        )
        assert 80.0 <= conf_full <= 100.0
        assert 0.0 <= bd_full["quality"] <= 30.0
        assert 0.0 <= bd_full["rule_match"] <= 50.0
        assert bd_full["stability"] == 20.0  # 1.0 * 20.0

        # 3. Open Palm with zero consensus
        conf_zero, bd_zero = recognizer.calculate_confidence(
            hand, features, "Open Palm", history_confidence=0.0
        )
        assert bd_zero["stability"] == 0.0
        assert conf_zero < conf_full

        # 4. Unknown gesture has lower rule match score
        pinky_only = create_mock_hand(False, False, False, False, True)
        features_pinky = recognizer.extract_features(pinky_only)
        conf_unk, bd_unk = recognizer.calculate_confidence(
            pinky_only, features_pinky, "Unknown", history_confidence=0.5
        )
        assert bd_unk["rule_match"] == 10.0

    def test_process_gesture_returns_gesture_result(self, recognizer):
        """Test process_gesture produces complete type-safe GestureResult."""
        hand = create_mock_hand(False, True, False, False, False)
        result = recognizer.process_gesture(hand, history_confidence=0.75)

        assert isinstance(result, GestureResult)
        assert result.gesture == "One Finger"
        assert result.raw_gesture == "One Finger"
        assert result.is_hand_detected is True
        assert 0.0 <= result.confidence <= 100.0
        assert result.features is not None
        assert result.confidence_breakdown is not None
        assert "quality" in result.confidence_breakdown
        assert "rule_match" in result.confidence_breakdown
        assert "stability" in result.confidence_breakdown

    def test_threshold_management(self, recognizer):
        """Test threshold accessors, setters, and updates."""
        default_ratio = recognizer.get_threshold("extension_ratio_index")
        assert default_ratio == 1.35

        recognizer.set_threshold("extension_ratio_index", 1.40)
        assert recognizer.get_threshold("extension_ratio_index") == 1.40

        recognizer.update_thresholds({
            "extension_ratio_middle": 1.42,
            "straightness_threshold": 0.90,
        })
        assert recognizer.get_threshold("extension_ratio_middle") == 1.42
        assert recognizer.get_threshold("straightness_threshold") == 0.90

        with pytest.raises(KeyError):
            recognizer.set_threshold("invalid_threshold_name", 0.5)

        all_thresh = recognizer.get_all_thresholds()
        assert isinstance(all_thresh, dict)
        assert "extension_ratio_index" in all_thresh
