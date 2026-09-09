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
    def test_failed_initialization_releases_camera_and_raises(self, mock_videocapture):
        """Test that failed camera initialization releases capture resource before raising RuntimeError."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_videocapture.return_value = mock_cap

        with pytest.raises(RuntimeError, match="Failed to open camera"):
            CameraCapture(camera_id=99)

        mock_cap.release.assert_called()

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

        camera = CameraCapture(camera_id=0, target_width=1280, target_height=720, resize_factor=0.5, force_vga=False)

        assert camera.get_frame_dimensions() == (640, 360)

        success, frame = camera.get_frame()
        assert success is True
        assert frame.shape == (360, 640, 3)  # height=360, width=640

    @patch("cv2.VideoCapture")
    def test_invalid_resize_factors(self, mock_videocapture):
        """Test rejection of 0, negative, non-numeric, or out-of-range resize factors."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_videocapture.return_value = mock_cap

        # Zero factor
        with pytest.raises(ValueError):
            CameraCapture(resize_factor=0.0)

        # Negative factor
        with pytest.raises(ValueError):
            CameraCapture(resize_factor=-0.5)

        # Non-numeric string factor
        with pytest.raises(TypeError):
            CameraCapture(resize_factor="invalid")

        # None factor
        with pytest.raises(TypeError):
            CameraCapture(resize_factor=None)

        # Out of bounds (> 1.0)
        with pytest.raises(ValueError):
            CameraCapture(resize_factor=1.5)

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

    @patch("cv2.VideoCapture")
    def test_mirror_setting_and_continuity(self, mock_videocapture):
        """Test mirror flipping flag and memory continuity preservation."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0, 5: 30.0}.get(prop, 0.0)
        # Non-symmetric gradient pattern to test mirror flip
        frame_pattern = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_pattern[:, :320] = 255  # Left half white
        mock_cap.read.return_value = (True, frame_pattern.copy())
        mock_videocapture.return_value = mock_cap

        # Default mirror is True
        camera = CameraCapture(camera_id=0, mirror=True)
        ret, frame = camera.get_frame()
        assert ret is True
        assert frame.flags.c_contiguous
        # When mirrored, right half should now be white
        assert np.all(frame[:, 320:] == 255)

        # Toggle mirror off
        camera.set_mirror(False)
        mock_cap.read.return_value = (True, frame_pattern.copy())
        ret, unmirrored = camera.get_frame()
        assert ret is True
        assert np.all(unmirrored[:, :320] == 255)

        # Invalid mirror type raises TypeError
        with pytest.raises(TypeError):
            camera.set_mirror("invalid")
        with pytest.raises(TypeError):
            CameraCapture(mirror=123)

    @patch("cv2.VideoCapture")
    def test_auto_contrast_setting_and_execution(self, mock_videocapture):
        """Test auto contrast setting and execution without error."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0, 5: 30.0}.get(prop, 0.0)
        fake_frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        mock_cap.read.return_value = (True, fake_frame)
        mock_videocapture.return_value = mock_cap

        camera = CameraCapture(camera_id=0, auto_contrast=True)
        ret, frame = camera.get_frame()
        assert ret is True
        assert frame.shape == (480, 640, 3)
        assert frame.flags.c_contiguous

        # Invalid auto_contrast type raises TypeError
        with pytest.raises(TypeError):
            camera.set_auto_contrast(None)
        with pytest.raises(TypeError):
            CameraCapture(auto_contrast="yes")

    @patch("cv2.VideoCapture")
    def test_force_vga_downscales_high_resolution_frames(self, mock_videocapture):
        """RC1: force_vga=True must downscale 1080p frames to 640x480 to prevent BlazePalm anchor mismatch."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # Driver reports it negotiated 1080p despite our 640x480 request
        mock_cap.get.side_effect = lambda prop: {3: 1920.0, 4: 1080.0, 5: 30.0}.get(prop, 0.0)
        hd_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, hd_frame)
        mock_videocapture.return_value = mock_cap

        camera = CameraCapture(camera_id=0, force_vga=True)
        ret, frame = camera.get_frame()
        assert ret is True
        # Frame must be downscaled to VGA safe dimensions
        assert frame.shape == (480, 640, 3)
        assert frame.flags.c_contiguous

    @patch("cv2.VideoCapture")
    def test_force_vga_false_preserves_high_resolution(self, mock_videocapture):
        """RC1: force_vga=False must leave high-resolution frames at their native size."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 1920.0, 4: 1080.0, 5: 30.0}.get(prop, 0.0)
        hd_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, hd_frame)
        mock_videocapture.return_value = mock_cap

        camera = CameraCapture(camera_id=0, force_vga=False)
        ret, frame = camera.get_frame()
        assert ret is True
        assert frame.shape[0] == 1080  # native height preserved
        assert frame.shape[1] == 1920  # native width preserved

    @patch("cv2.VideoCapture")
    def test_bgra_alpha_strip(self, mock_videocapture):
        """RC2: BGRA (4-channel) frames from webcam drivers must be stripped to 3-channel BGR."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0, 5: 30.0}.get(prop, 0.0)
        # Simulate a 4-channel BGRA frame (common on Linux UVC webcams)
        bgra_frame = np.full((480, 640, 4), 200, dtype=np.uint8)
        bgra_frame[:, :, 3] = 255  # alpha = fully opaque
        mock_cap.read.return_value = (True, bgra_frame)
        mock_videocapture.return_value = mock_cap

        camera = CameraCapture(camera_id=0, force_vga=False)
        ret, frame = camera.get_frame()
        assert ret is True
        assert frame.shape[2] == 3, "Alpha channel must be stripped before returning frame"
        assert frame.flags.c_contiguous

    @patch("cv2.VideoCapture")
    def test_force_vga_invalid_type_raises_type_error(self, mock_videocapture):
        """RC1: Non-boolean force_vga must raise TypeError."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_videocapture.return_value = mock_cap
        with pytest.raises(TypeError, match="force_vga must be a boolean"):
            CameraCapture(force_vga=1)
