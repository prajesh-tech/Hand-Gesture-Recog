"""
Camera module for capturing video frames from webcam.
Handles camera initialization, frame capture, and graceful error handling.
"""

import cv2
import time
from typing import Optional, Tuple


class CameraCapture:
    """Wrapper for OpenCV VideoCapture with error handling and FPS tracking."""
    
    def __init__(self, camera_id: int = 0, target_width: int = 640, target_height: int = 480, 
                 target_fps: int = 30, resize_factor: float = 1.0):
        """
        Initialize camera capture.
        
        Args:
            camera_id: Camera device ID (default 0 for primary webcam)
            target_width: Target frame width
            target_height: Target frame height
            target_fps: Target frames per second (for reference only; actual depends on hardware)
            resize_factor: Factor to resize frames (0.5 = half size, 1.0 = original)
        
        Raises:
            RuntimeError: If camera cannot be initialized
        """
        self.camera_id = camera_id
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self.resize_factor = resize_factor
        
        self.cap = None
        self.frame_count = 0
        self.last_timestamp = time.time()
        self.fps = 0.0
        
        self._initialize_camera()
    
    def _initialize_camera(self) -> None:
        """Initialize camera device and set properties."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera (ID: {self.camera_id}). Check camera connection.")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Read actual properties
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
        except Exception as e:
            raise RuntimeError(f"Camera initialization failed: {str(e)}")
    
    def get_frame(self) -> Tuple[bool, Optional[any]]:
        """
        Capture next frame from camera.
        
        Returns:
            Tuple of (success: bool, frame: numpy.ndarray or None)
            - success: True if frame was captured, False if camera dropped or error
            - frame: BGR frame (resized if resize_factor < 1.0), or None if failed
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
            print(f"Error capturing frame: {str(e)}")
            return False, None
    
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Get current frame dimensions (width, height) after any resizing."""
        if self.cap is None:
            return int(self.target_width * self.resize_factor), int(self.target_height * self.resize_factor)
        
        return int(self.actual_width * self.resize_factor), int(self.actual_height * self.resize_factor)

    
    def get_fps(self) -> float:
        """Get measured FPS from recent frames."""
        return self.fps
    
    def set_resize_factor(self, factor: float) -> None:
        """Update frame resize factor (e.g., 0.5 for half-size)."""
        self.resize_factor = max(0.1, min(1.0, factor))  # Clamp between 0.1 and 1.0
    
    def is_open(self) -> bool:
        """Check if camera is still open and accessible."""
        if self.cap is None:
            return False
        return self.cap.isOpened()
    
    def release(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
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
