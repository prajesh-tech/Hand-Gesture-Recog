"""
Camera module for capturing video frames from webcam.
Handles camera initialization, frame capture, resize validation, and graceful error handling.
"""

import time
from typing import Optional, Tuple
import cv2
import numpy as np


class CameraCapture:
    """Wrapper for OpenCV VideoCapture with error handling, resize factor validation, and FPS tracking."""

    def __init__(
        self,
        camera_id: int = 0,
        target_width: int = 640,
        target_height: int = 480,
        target_fps: int = 30,
        resize_factor: float = 1.0,
    ):
        self.cap = None
        self.camera_id = camera_id
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps

        self.resize_factor = self._validate_resize_factor(resize_factor)

        self.frame_count = 0
        self.last_timestamp = time.time()
        self.fps = 0.0

        self.actual_width = target_width
        self.actual_height = target_height
        self.actual_fps = float(target_fps)

        self._initialize_camera()

    @staticmethod
    def _validate_resize_factor(factor: float) -> float:
        """Validate resize_factor: must be float/int in range [0.1, 1.0]."""
        if factor is None:
            raise TypeError("resize_factor cannot be None")
        if isinstance(factor, bool) or not isinstance(factor, (int, float)):
            raise TypeError(f"resize_factor must be a numeric value, got {type(factor).__name__}")
        float_factor = float(factor)
        if float_factor <= 0.0:
            raise ValueError(f"resize_factor must be positive (> 0), got {float_factor}")
        if float_factor < 0.1 or float_factor > 1.0:
            raise ValueError(f"resize_factor must be between 0.1 and 1.0, got {float_factor}")
        return float_factor

    def set_resize_factor(self, factor: float) -> None:
        """Update frame resize factor."""
        self.resize_factor = self._validate_resize_factor(factor)

    def _initialize_camera(self) -> None:
        """Initialize camera device and set properties. Releases capture object on failure."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                self.release()
                raise RuntimeError(
                    f"Failed to open camera (ID: {self.camera_id}). Check camera connection."
                )

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # Read actual properties
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.target_width
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.target_height
            self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS) or float(self.target_fps)

        except Exception as e:
            self.release()
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Camera initialization failed: {e!s}") from e

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture next frame from camera.

        Returns:
            Tuple of (success: bool, frame: numpy.ndarray or None)
            - success: True if frame was captured, False if camera dropped or error
            - frame: BGR frame (resized according to resize_factor), or None if failed
        """
        if self.cap is None:
            return False, None

        try:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                return False, None

            # Apply resizing if needed
            if self.resize_factor < 1.0:
                new_width = int(frame.shape[1] * self.resize_factor)
                new_height = int(frame.shape[0] * self.resize_factor)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # Update FPS counter
            self.frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.last_timestamp

            if elapsed >= 1.0:  # Update FPS every second
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.last_timestamp = current_time

            return True, frame

        except Exception as e:
            print(f"Error capturing frame: {e!s}")
            return False, None

    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Get current processed frame dimensions (width, height) after resizing."""
        return int(self.actual_width * self.resize_factor), int(self.actual_height * self.resize_factor)

    def get_fps(self) -> float:
        """Get measured FPS from recent frames."""
        return self.fps

    def is_open(self) -> bool:
        """Check if camera is open and accessible."""
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        """Release camera resources safely."""
        cap = getattr(self, "cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            self.cap = None

    def __del__(self):
        """Ensure camera is released when object is destroyed."""
        self.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release camera."""
        self.release()
