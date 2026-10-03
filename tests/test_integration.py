"""
Integration tests for the MediaPipe 3D Landmark Strategy 1 pipeline.
Tests end-to-end processing, state pipeline dynamics, temporal consensus,
and robustness without physical webcam or external network dependencies.
"""

import os
import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.camera import CameraCapture
from src.main import HandGestureApp
from src.results import DetectionFrameResult
from tests.test_gesture_recognition import create_mock_hand


class TestIntegrationStrategy1:
    """Integration test suite for Strategy 1 pipeline."""

    @pytest.fixture
    def app_pipeline(self):
        """Set up headless HandGestureApp pipeline with mocked camera."""
        with patch.object(CameraCapture, "__init__", lambda self, *args, **kwargs: None):
            app = HandGestureApp(history_buffer_size=4, consensus_threshold=3)
            app.camera = MagicMock()
            app.camera.get_fps.return_value = 30.0
            yield app
            app.cleanup()

    def test_full_pipeline_hand_detection_and_consensus(self, app_pipeline):
        """Test full pipeline flow from raw frame to stable gesture and action."""
        app = app_pipeline
        mock_frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        app.camera.get_frame.return_value = (True, mock_frame)

        # Mock Open Palm landmarks
        open_palm_lms = create_mock_hand(True, True, True, True, True)

        with patch.object(app.detector, "process_frame") as mock_proc:
            mock_proc.return_value = (open_palm_lms, mock_frame.copy())

            # Frame 1: Consensus not yet reached (threshold=3)
            res1 = app.process_current_frame()
            assert isinstance(res1, DetectionFrameResult)
            assert res1.gesture_res.raw_gesture == "Open Palm"
            assert res1.history_res.smoothed_gesture is None
            assert res1.history_res.action == ""
            assert res1.gesture_res.confidence > 0.0

            # Frame 2: Still stabilizing
            res2 = app.process_current_frame()
            assert res2.history_res.smoothed_gesture is None

            # Frame 3: Consensus reached!
            res3 = app.process_current_frame()
            assert res3.history_res.smoothed_gesture == "Open Palm"
            assert res3.history_res.action == "START"
            # Stability score should now be high (3 of 3 in history = 1.0 -> +20 pts)
            assert res3.gesture_res.confidence_breakdown["stability"] == 20.0
            assert res3.gesture_res.confidence >= 80.0

    def test_gesture_transition_dynamics(self, app_pipeline):
        """Test transitioning between gestures requires new consensus window."""
        app = app_pipeline
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        app.camera.get_frame.return_value = (True, mock_frame)

        fist_lms = create_mock_hand(False, False, False, False, False)
        two_fingers_lms = create_mock_hand(False, True, True, False, False)

        with patch.object(app.detector, "process_frame") as mock_proc:
            # Feed 3 Fist frames to establish consensus
            mock_proc.return_value = (fist_lms, mock_frame.copy())
            for _ in range(3):
                res = app.process_current_frame()
            assert res.history_res.smoothed_gesture == "Fist"
            assert res.history_res.action == "STOP"

            # Switch to Two Fingers: frame 1 should NOT switch to Two Fingers
            # (Buffer is ['Fist', 'Fist', 'Fist', 'Two Fingers'] - Fist still has 3 votes)
            mock_proc.return_value = (two_fingers_lms, mock_frame.copy())
            res_trans1 = app.process_current_frame()
            assert res_trans1.gesture_res.raw_gesture == "Two Fingers"
            assert res_trans1.history_res.smoothed_gesture != "Two Fingers"

            # Frame 2 of Two Fingers: Buffer is ['Fist', 'Fist', 'Two Fingers', 'Two Fingers']
            # Neither has 3 votes, consensus drops to None
            res_trans2 = app.process_current_frame()
            assert res_trans2.history_res.smoothed_gesture is None

            # Frame 3 of Two Fingers: Buffer is ['Fist', 'Two Fingers', 'Two Fingers', 'Two Fingers']
            # Two Fingers reaches 3 votes -> consensus reached!
            res_stable = app.process_current_frame()
            assert res_stable.history_res.smoothed_gesture == "Two Fingers"
            assert res_stable.history_res.action == "NEXT"

    def test_pipeline_no_hand_detected(self, app_pipeline):
        """Test pipeline behavior when no hand is present."""
        app = app_pipeline
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        app.camera.get_frame.return_value = (True, mock_frame)

        with patch.object(app.detector, "process_frame") as mock_proc:
            mock_proc.return_value = (None, mock_frame.copy())
            res = app.process_current_frame()

            assert res.landmark_res.is_hand_detected is False
            assert res.gesture_res.gesture is None
            assert res.gesture_res.raw_gesture is None
            assert res.gesture_res.confidence == 0.0
            assert res.history_res.smoothed_gesture is None
            assert res.history_res.action == ""

    def test_pipeline_overlay_rendering(self, app_pipeline):
        """Test that HUD overlay generates valid image array without exceptions."""
        app = app_pipeline
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        app.camera.get_frame.return_value = (True, mock_frame)

        one_finger_lms = create_mock_hand(False, True, False, False, False)
        with patch.object(app.detector, "process_frame") as mock_proc:
            mock_proc.return_value = (one_finger_lms, mock_frame.copy())
            # Run 3 frames for consensus
            for _ in range(3):
                diag = app.process_current_frame()

            overlay = app.render_overlay(diag)
            assert isinstance(overlay, np.ndarray)
            assert overlay.shape == (480, 640, 3)

    def test_pipeline_camera_dropped_returns_none(self, app_pipeline):
        """Test graceful handling when camera stream drops."""
        app = app_pipeline
        app.camera.get_frame.return_value = (False, None)
        res = app.process_current_frame()
        assert res is None

    def test_pipeline_bypass_classifier_mode(self, app_pipeline, capsys):
        """Test classifier bypass mode prints HAND DETECTED and renders bypass overlay."""
        app = app_pipeline
        app.bypass_classifier = True
        app.draw_landmark_labels = True

        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        app.camera.get_frame.return_value = (True, mock_frame)

        open_palm_lms = create_mock_hand(True, True, True, True, True)
        with patch.object(app.detector, "process_frame") as mock_proc:
            mock_proc.return_value = (open_palm_lms, mock_frame.copy())
            res = app.process_current_frame()

            assert res is not None
            assert res.landmark_res.is_hand_detected is True
            assert res.gesture_res.gesture == "Bypassed"
            assert res.gesture_res.confidence == 100.0

            captured = capsys.readouterr()
            assert "HAND DETECTED" in captured.out
            assert "Landmarks: 21" in captured.out

            overlay = app.render_overlay(res)
            assert isinstance(overlay, np.ndarray)

        # Test NO HAND branch in bypass mode
        with patch.object(app.detector, "process_frame") as mock_proc:
            mock_proc.return_value = (None, mock_frame.copy())
            res = app.process_current_frame()
            assert res.landmark_res.is_hand_detected is False

            captured = capsys.readouterr()
            assert "NO HAND" in captured.out
