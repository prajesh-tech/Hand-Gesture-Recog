"""
Skin detection module using HSV color space thresholding.
Provides HSV-based skin color segmentation with morphological cleanup.
"""

from typing import Tuple  # noqa: F401, UP035

import cv2
import numpy as np


class SkinDetector:
    """HSV-based skin color detection with configurable thresholds."""
    
    # Default HSV ranges for typical skin tone (tunable based on user calibration)
    DEFAULT_LOWER_HSV = np.array([0, 10, 60], dtype=np.uint8)
    DEFAULT_UPPER_HSV = np.array([20, 150, 255], dtype=np.uint8)
    
    def __init__(self, lower_hsv: np.ndarray | None = None, 
                 upper_hsv: np.ndarray | None = None,
                 morphology_kernel_size: int = 5):
        """
        Initialize skin detector.
        
        Args:
            lower_hsv: Lower HSV threshold (3-element uint8 array [H, S, V])
            upper_hsv: Upper HSV threshold (3-element uint8 array [H, S, V])
            morphology_kernel_size: Size of morphological kernel (odd number)
        """
        self.lower_hsv = lower_hsv if lower_hsv is not None else self.DEFAULT_LOWER_HSV.copy()
        self.upper_hsv = upper_hsv if upper_hsv is not None else self.DEFAULT_UPPER_HSV.copy()
        
        # Ensure morphology kernel size is odd
        if morphology_kernel_size % 2 == 0:
            morphology_kernel_size += 1
        
        self.morphology_kernel_size = morphology_kernel_size
        self.morphology_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (morphology_kernel_size, morphology_kernel_size)
        )
    
    def set_hsv_range(self, lower_hsv: np.ndarray, upper_hsv: np.ndarray) -> None:
        """
        Update HSV thresholds.
        
        Args:
            lower_hsv: Lower bound [H, S, V] (each 0-255)
            upper_hsv: Upper bound [H, S, V] (each 0-255)
        """
        self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
    
    def detect_skin(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Detect skin regions in BGR frame.
        
        Args:
            frame_bgr: Input frame in BGR color space
        
        Returns:
            Binary mask where white (255) indicates detected skin, black (0) otherwise
        """
        # Convert BGR to HSV
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # Apply HSV threshold to create binary mask
        mask = cv2.inRange(frame_hsv, self.lower_hsv, self.upper_hsv)
        
        return mask
    
    def apply_morphology(self, mask: np.ndarray, operation: str = 'open') -> np.ndarray:
        """
        Apply morphological operations to clean up binary mask.
        Removes noise and fills small holes.
        
        Args:
            mask: Binary mask
            operation: 'open' (remove small noise), 'close' (fill holes), 'both' (open then close)
        
        Returns:
            Cleaned binary mask
        """
        if operation == 'open':
            return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morphology_kernel, iterations=2)
        elif operation == 'close':
            return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morphology_kernel, iterations=2)
        elif operation == 'both':
            # Open first (remove noise), then close (fill holes)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morphology_kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morphology_kernel, iterations=1)
            return mask
        else:
            return mask
    
    def detect_and_clean(self, frame_bgr: np.ndarray, morphology_op: str = 'both') -> np.ndarray:
        """
        Convenience method: detect skin and apply morphological cleanup in one call.
        
        Args:
            frame_bgr: Input frame in BGR color space
            morphology_op: Morphological operation ('open', 'close', or 'both')
        
        Returns:
            Cleaned binary skin mask
        """
        mask = self.detect_skin(frame_bgr)
        mask = self.apply_morphology(mask, operation=morphology_op)
        return mask
    
    def get_hsv_range(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current HSV thresholds."""
        return self.lower_hsv.copy(), self.upper_hsv.copy()
    
    @staticmethod
    def extract_hsv_from_region(frame_bgr: np.ndarray, region_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute HSV range from pixels in a specific region (for calibration).
        
        Args:
            frame_bgr: Input frame in BGR
            region_mask: Binary mask indicating region of interest (white = include)
        
        Returns:
            Tuple of (lower_hsv, upper_hsv) representing the HSV range of the region
        """
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # Extract HSV values for masked region
        hsv_values = frame_hsv[region_mask > 0]
        
        if len(hsv_values) == 0:
            # Empty region, return defaults
            return SkinDetector.DEFAULT_LOWER_HSV.copy(), SkinDetector.DEFAULT_UPPER_HSV.copy()
        
        # Compute min and max for each channel
        lower = hsv_values.min(axis=0).astype(np.uint8)
        upper = hsv_values.max(axis=0).astype(np.uint8)
        
        # ✨ CRITICAL FIX: Clamp Hue to 0-180 range (OpenCV convention)
        # If Hue values seem out of range, they may have been uint8-wrapped
        if lower[0] > 180:
            lower[0] = max(0, lower[0] - 256)
        if upper[0] > 180:
            upper[0] = min(180, upper[0])
        
        # Add small margin for tolerance (but respect OpenCV bounds)
        lower = np.array([
            max(0, lower[0] - 10) if lower[0] <= 180 else 0,
            max(0, lower[1] - 10),
            max(0, lower[2] - 10)
        ], dtype=np.uint8)
        
        upper = np.array([
            min(180, upper[0] + 10),  # Hue: 0-180
            min(255, upper[1] + 10),  # Saturation: 0-255
            min(255, upper[2] + 10)   # Value: 0-255
        ], dtype=np.uint8)
        
        return lower, upper
