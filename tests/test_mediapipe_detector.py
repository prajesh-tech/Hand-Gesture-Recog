"""
Unit tests for MediaPipeDetector class.
Tests parameter initialization, input validation, mocked detection, and clean teardown.
Runs completely offline without physical camera access.
"""

import os
import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mediapipe_detector import MediaPipeDetector
from src.results import LandmarkDetectionResult


class TestMediaPipeDetector:
    """Test suite for MediaPipeDetector component."""

    def test_default_initialization(self):
        """Test default detector initialization."""
        detector = MediaPipeDetector()
        assert detector.min_detection_confidence == 0.5
        assert detector.min_tracking_confidence == 0.5
        assert detector.max_num_hands == 1
        assert detector.static_image_mode is False
        detector.close()

    def test_custom_initialization(self):
        """Test custom parameter configuration."""
        detector = MediaPipeDetector(
            min_detection_confidence=0.85,
            min_tracking_confidence=0.6,
            max_num_hands=2,
            static_image_mode=True,
        )
        assert detector.min_detection_confidence == 0.85
        assert detector.min_tracking_confidence == 0.6
        assert detector.max_num_hands == 2
        assert detector.static_image_mode is True
        detector.close()

    def test_invalid_parameters_raise_value_error(self):
        """Test that invalid thresholds or hand counts raise ValueError."""
        with pytest.raises(ValueError):
            MediaPipeDetector(min_detection_confidence=-0.1)
        with pytest.raises(ValueError):
            MediaPipeDetector(min_detection_confidence=1.5)
        with pytest.raises(ValueError):
            MediaPipeDetector(min_tracking_confidence=-0.5)
        with pytest.raises(ValueError):
            MediaPipeDetector(max_num_hands=0)

    def test_missing_model_auto_download_failure(self, tmp_path):
        """Test that failure during model auto-download raises FileNotFoundError with helpful instructions."""
        fake_model = str(tmp_path / "models" / "hand_landmarker.task")
        with patch("os.path.isfile", return_value=False), \
             patch("urllib.request.urlretrieve", side_effect=OSError("Network down")):
            with pytest.raises(FileNotFoundError, match="Auto-download failed"):
                MediaPipeDetector(model_path=fake_model)

    def test_input_validation_on_invalid_frames(self):
        """Test defensive input validation on empty, None, or ill-shaped frames."""
        detector = MediaPipeDetector()
        try:
            # None frame
            with pytest.raises(TypeError, match="Frame cannot be None"):
                detector.validate_frame(None)

            # Non-ndarray
            with pytest.raises(TypeError, match="Frame must be a numpy.ndarray"):
                detector.validate_frame("not an array")

            # Empty array
            with pytest.raises(ValueError, match="non-empty"):
                detector.validate_frame(np.zeros((0, 0, 3), dtype=np.uint8))

            # 2D grayscale image (missing channels)
            with pytest.raises(ValueError, match="non-empty 3-channel"):
                detector.validate_frame(np.zeros((100, 100), dtype=np.uint8))

            # 4-channel image
            with pytest.raises(ValueError, match="exactly 3 color channels"):
                detector.validate_frame(np.zeros((100, 100, 4), dtype=np.uint8))

            # Non-uint8 dtype (e.g. float32)
            with pytest.raises(TypeError, match="Frame dtype must be uint8"):
                detector.validate_frame(np.zeros((100, 100, 3), dtype=np.float32))
        finally:
            detector.close()

    def test_process_frame_no_hand_detected(self):
        """Test process_frame when MediaPipe returns no hand landmarks."""
        detector = MediaPipeDetector()
        try:
            with patch.object(detector.hands, "detect_for_video") as mock_detect:
                mock_result = MagicMock()
                mock_result.hand_landmarks = None
                mock_detect.return_value = mock_result

                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                landmarks, annotated = detector.process_frame(blank_frame)

                assert landmarks is None
                assert isinstance(annotated, np.ndarray)
                assert annotated.shape == blank_frame.shape
        finally:
            detector.close()

    def test_process_frame_hand_detected(self):
        """Test process_frame with mock detected hand landmarks."""
        detector = MediaPipeDetector()
        try:
            mock_landmarks = []
            for i in range(21):
                mock_lm = MagicMock()
                mock_lm.x = 0.5 + i * 0.01
                mock_lm.y = 0.5 + i * 0.01
                mock_lm.z = -0.05 + i * 0.001
                mock_landmarks.append(mock_lm)

            mock_result = MagicMock()
            mock_result.hand_landmarks = [mock_landmarks]

            with patch.object(detector.hands, "detect_for_video", return_value=mock_result):
                test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
                landmarks, annotated = detector.process_frame(test_frame)

                assert landmarks is not None
                assert len(landmarks) == 21
                assert landmarks[0]["id"] == 0
                assert isinstance(landmarks[0]["x"], float)
                assert isinstance(landmarks[0]["y"], float)
                assert isinstance(landmarks[0]["z"], float)
                assert annotated.shape == test_frame.shape

                # Test process_frame_to_result wrapper
                res = detector.process_frame_to_result(test_frame)
                assert isinstance(res, LandmarkDetectionResult)
                assert res.is_hand_detected is True
                assert len(res.landmarks) == 21
        finally:
            detector.close()

    def test_context_manager_and_close(self):
        """Test context manager lifecycle and close() method."""
        with MediaPipeDetector() as detector:
            assert detector.hands is not None
        # Verify close method can be called multiple times without error
        detector.close()

    def test_process_frame_handles_non_contiguous_input(self):
        """Test that process_frame handles non-contiguous (e.g. flipped or sliced) arrays gracefully."""
        detector = MediaPipeDetector()
        try:
            # Create a non-contiguous slice/flip
            frame = np.full((480, 640, 3), 128, dtype=np.uint8)
            flipped_frame = np.fliplr(frame)  # Not C-contiguous
            assert not flipped_frame.flags.c_contiguous

            with patch.object(detector.hands, "detect_for_video") as mock_detect:
                mock_result = MagicMock()
                mock_result.hand_landmarks = None
                mock_detect.return_value = mock_result

                lms, ann = detector.process_frame(flipped_frame)
                assert lms is None
                assert ann.shape == flipped_frame.shape
        finally:
            detector.close()

    def test_enhance_contrast_functionality_and_validation(self):
        """Test CLAHE contrast enhancement utility and input validation."""
        test_frame = np.full((100, 100, 3), 100, dtype=np.uint8)
        enhanced = MediaPipeDetector.enhance_contrast(test_frame)
        assert isinstance(enhanced, np.ndarray)
        assert enhanced.shape == test_frame.shape
        assert enhanced.dtype == np.uint8

        # Input validation
        with pytest.raises(TypeError):
            MediaPipeDetector.enhance_contrast(None)
        with pytest.raises(TypeError):
            MediaPipeDetector.enhance_contrast("invalid")
        with pytest.raises(ValueError):
            MediaPipeDetector.enhance_contrast(np.zeros((0, 0, 3), dtype=np.uint8))
        with pytest.raises(ValueError):
            MediaPipeDetector.enhance_contrast(np.zeros((100, 100), dtype=np.uint8))

    def test_wrist_boundary_warning_emitted_when_near_edge(self, capsys):
        """RC3: process_frame must emit a console warning when wrist (LM0) is within 10% of any frame edge."""
        detector = MediaPipeDetector()
        try:
            # Construct 21 mock landmarks with wrist at x=0.02 — within the 10% edge margin
            mock_landmarks = []
            for i in range(21):
                mock_lm = MagicMock()
                mock_lm.x = 0.02 if i == 0 else (0.5 + i * 0.01)  # wrist very close to left edge
                mock_lm.y = 0.5
                mock_lm.z = -0.05
                mock_landmarks.append(mock_lm)

            mock_result = MagicMock()
            mock_result.hand_landmarks = [mock_landmarks]

            with patch.object(detector.hands, "detect_for_video", return_value=mock_result):
                test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
                landmarks, _ = detector.process_frame(test_frame)

            assert landmarks is not None
            assert landmarks[0]["x"] == pytest.approx(0.02)

            captured = capsys.readouterr()
            assert "Wrist (LM0) near frame edge" in captured.out, (
                "Expected boundary proximity warning in stdout when wrist is within 10% of edge"
            )
        finally:
            detector.close()
