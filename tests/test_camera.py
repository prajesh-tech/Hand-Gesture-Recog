"""
Unit tests for CameraCapture module with mocked OpenCV VideoCapture.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.camera import CameraCapture


class TestCameraCapture:
    """Test suite for CameraCapture class."""

    @patch("cv2.VideoCapture")
    def test_successful_initialization(self, mock_videocapture):
        """Test successful camera initialization."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0, 5: 30.0}.get(prop, 0.0)
        mock_videocapture.return_value = mock_cap

        camera = CameraCapture(camera_id=0, target_width=640, target_height=480)

        assert camera.is_open() is True
        assert camera.get_frame_dimensions() == (640, 480)
        mock_videocapture.assert_called_once_with(0)

    @patch("cv2.VideoCapture")
    def test_failed_initialization_raises_runtime_error(self, mock_videocapture):
        """Test that failed camera initialization raises RuntimeError."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_videocapture.return_value = mock_cap

        with pytest.raises(RuntimeError, match="Failed to open camera"):
            CameraCapture(camera_id=99)

    @patch("cv2.VideoCapture")
    def test_frame_acquisition(self, mock_videocapture):
        """Test retrieving a frame from camera."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0, 5: 30.0}.get(prop, 0.0)
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, fake_frame)
        mock_videocapture.return_value = mock_cap

        camera = CameraCapture(camera_id=0)
        success, frame = camera.get_frame()

        assert success is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    @patch("cv2.VideoCapture")
    def test_resize_behavior_and_dimensions(self, mock_videocapture):
        """Test frame resizing and get_frame_dimensions with resize_factor."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 1280.0, 4: 720.0, 5: 30.0}.get(prop, 0.0)
        fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, fake_frame)
        mock_videocapture.return_value = mock_cap

        camera = CameraCapture(camera_id=0, target_width=1280, target_height=720, resize_factor=0.5)

        assert camera.get_frame_dimensions() == (640, 360)

        success, frame = camera.get_frame()
        assert success is True
        assert frame.shape == (360, 640, 3)  # height=360, width=640

    @patch("cv2.VideoCapture")
    def test_release_and_cleanup(self, mock_videocapture):
        """Test camera release resource cleanup."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0, 5: 30.0}.get(prop, 0.0)
        mock_videocapture.return_value = mock_cap

        camera = CameraCapture(camera_id=0)
        camera.release()

        assert camera.is_open() is False
        mock_cap.release.assert_called_once()
