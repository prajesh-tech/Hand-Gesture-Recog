"""
Improved skin detection with statistical HSV model and robust calibration.
Uses percentile-based bounds and handles hue wrap-around correctly.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


class SkinDetectorImproved:
    """
    Enhanced skin detector with:
    - Robust calibration using percentiles instead of min/max
    - Statistical distance-based skin detection (Mahalanobis)
    - Proper hue wrap-around handling
    - Configurable morphology
    """
    
    # Default HSV for typical skin tone
    DEFAULT_LOWER_HSV = np.array([0, 10, 60], dtype=np.uint8)
    DEFAULT_UPPER_HSV = np.array([20, 150, 255], dtype=np.uint8)
    
    def __init__(self, lower_hsv: Optional[np.ndarray] = None,
                 upper_hsv: Optional[np.ndarray] = None,
                 morphology_kernel_size: int = 5,
                 morphology_strength: str = 'normal'):
        """
        Initialize improved skin detector.
        
        Args:
            lower_hsv: Lower HSV threshold
            upper_hsv: Upper HSV threshold
            morphology_kernel_size: Odd integer for kernel size
            morphology_strength: 'light', 'normal', 'strong'
        """
        self.lower_hsv = lower_hsv if lower_hsv is not None else self.DEFAULT_LOWER_HSV.copy()
        self.upper_hsv = upper_hsv if upper_hsv is not None else self.DEFAULT_UPPER_HSV.copy()
        
        # Morphology config
        if morphology_kernel_size % 2 == 0:
            morphology_kernel_size += 1
        self.morphology_kernel_size = morphology_kernel_size
        self.morphology_strength = morphology_strength
        
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morphology_kernel_size, morphology_kernel_size)
        )
        
        # Statistical model (populated during calibration)
        self.skin_hsv_mean = None
        self.skin_hsv_cov = None
        self.use_statistical_model = False
        
        # Diagnostics
        self.last_calibration_stats = {}
    
    def detect_skin(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Detect skin using rectangular HSV threshold.
        
        Args:
            frame_bgr: Input frame in BGR
        
        Returns:
            Binary mask
        """
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # Handle hue wrap-around
        mask = self._apply_hsv_range(frame_hsv, self.lower_hsv, self.upper_hsv)
        
        return mask
    
    def detect_skin_statistical(self, frame_bgr: np.ndarray, distance_threshold: float = 2.5) -> np.ndarray:
        """
        Detect skin using Mahalanobis distance from calibrated mean.
        Only works if calibration has been done.
        
        Args:
            frame_bgr: Input frame in BGR
            distance_threshold: Mahalanobis distance threshold
        
        Returns:
            Binary mask
        """
        if not self.use_statistical_model or self.skin_hsv_mean is None:
            # Fall back to rectangular threshold
            return self.detect_skin(frame_bgr)
        
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, w = frame_hsv.shape[:2]
        
        # Reshape to list of HSV values
        pixels = frame_hsv.reshape(-1, 3).astype(np.float32)
        
        # Compute Mahalanobis distance
        try:
            # Add small regularization to covariance
            cov_reg = self.skin_hsv_cov + np.eye(3) * 0.1
            cov_inv = np.linalg.inv(cov_reg)
            
            diff = pixels - self.skin_hsv_mean
            distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
            
            # Create mask
            mask = (distances < distance_threshold).astype(np.uint8) * 255
            mask = mask.reshape(h, w)
            
            return mask
        except:
            # If Mahalanobis fails, fall back
            return self.detect_skin(frame_bgr)
    
    def apply_morphology(self, mask: np.ndarray, operation: str = 'both') -> np.ndarray:
        """
        Apply morphological operations.
        
        Args:
            mask: Binary mask
            operation: 'open', 'close', 'both', or 'none'
        
        Returns:
            Cleaned mask
        """
        if operation == 'none' or operation is None:
            return mask
        
        # Strength config
        if self.morphology_strength == 'light':
            open_iter, close_iter = 1, 0
        elif self.morphology_strength == 'normal':
            open_iter, close_iter = 2, 1
        else:  # strong
            open_iter, close_iter = 2, 2
        
        if operation in ['open', 'both']:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=open_iter)
        
        if operation in ['close', 'both']:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=close_iter)
        
        return mask
    
    def detect_and_clean(self, frame_bgr: np.ndarray, morphology_op: str = 'both',
                        use_statistical: bool = False) -> np.ndarray:
        """Detect skin and clean in one call."""
        if use_statistical and self.use_statistical_model:
            mask = self.detect_skin_statistical(frame_bgr)
        else:
            mask = self.detect_skin(frame_bgr)
        
        mask = self.apply_morphology(mask, operation=morphology_op)
        return mask
    
    def set_hsv_range(self, lower_hsv: np.ndarray, upper_hsv: np.ndarray) -> None:
        """Set HSV thresholds."""
        self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        self.use_statistical_model = False
    
    def get_hsv_range(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current HSV thresholds."""
        return self.lower_hsv.copy(), self.upper_hsv.copy()
    
    def _apply_hsv_range(self, frame_hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """
        Apply HSV range, handling hue wrap-around.
        
        Args:
            frame_hsv: Frame in HSV color space
            lower: Lower HSV bounds
            upper: Upper HSV bounds
        
        Returns:
            Binary mask
        """
        # Check if hue has wrap-around (lower_hue > upper_hue)
        if lower[0] > upper[0]:
            # Hue wraps: match lower[0]→180 OR 0→upper[0]
            mask1 = cv2.inRange(frame_hsv,
                               np.array([lower[0], lower[1], lower[2]]),
                               np.array([180, upper[1], upper[2]]))
            mask2 = cv2.inRange(frame_hsv,
                               np.array([0, lower[1], lower[2]]),
                               np.array([upper[0], upper[1], upper[2]]))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # Normal range: use inRange directly
            mask = cv2.inRange(frame_hsv, lower, upper)
        
        return mask
    
    @staticmethod
    def calibrate_from_samples(sample_frames: list, morph_kernel_size: int = 5) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Robust calibration from multiple sample frames using percentiles.
        
        Args:
            sample_frames: List of BGR frames (should contain hand in center region)
            morph_kernel_size: Kernel size for center extraction
        
        Returns:
            Tuple of (lower_hsv, upper_hsv, diagnostics_dict)
        """
        if not sample_frames or len(sample_frames) == 0:
            return (SkinDetectorImproved.DEFAULT_LOWER_HSV.copy(),
                   SkinDetectorImproved.DEFAULT_UPPER_HSV.copy(),
                   {"error": "No samples provided"})
        
        diagnostics = {}
        all_hsv_values = []
        center_hsv_values = []
        
        # Extract HSV from center region of each frame
        for frame in sample_frames:
            h, w = frame.shape[:2]
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Center region (inner 60% of frame)
            cy, cx = h // 2, w // 2
            region_h, region_w = int(h * 0.3), int(w * 0.3)
            y1, y2 = max(0, cy - region_h), min(h, cy + region_h)
            x1, x2 = max(0, cx - region_w), min(w, cx + region_w)
            
            center_region = frame_hsv[y1:y2, x1:x2]
            center_pixels = center_region.reshape(-1, 3)
            
            # Filter out dark/desaturated pixels (likely background)
            # Keep only pixels with S > 20 and V > 40
            valid_mask = (center_pixels[:, 1] > 20) & (center_pixels[:, 2] > 40)
            valid_pixels = center_pixels[valid_mask]
            
            center_hsv_values.extend(valid_pixels)
            all_hsv_values.extend(center_pixels)
        
        diagnostics['total_samples'] = len(sample_frames)
        diagnostics['total_pixels'] = len(all_hsv_values)
        diagnostics['valid_pixels'] = len(center_hsv_values)
        
        if len(center_hsv_values) == 0:
            diagnostics['error'] = 'No valid pixels after filtering'
            return (SkinDetectorImproved.DEFAULT_LOWER_HSV.copy(),
                   SkinDetectorImproved.DEFAULT_UPPER_HSV.copy(),
                   diagnostics)
        
        center_hsv_array = np.array(center_hsv_values, dtype=np.float32)
        valid_percentage = 100.0 * len(center_hsv_values) / len(all_hsv_values)
        diagnostics['valid_percentage'] = valid_percentage
        
        # Use percentiles instead of min/max for robustness
        lower_percentile = 5   # 5th percentile
        upper_percentile = 95  # 95th percentile
        
        lower = np.percentile(center_hsv_array, lower_percentile, axis=0).astype(np.uint8)
        upper = np.percentile(center_hsv_array, upper_percentile, axis=0).astype(np.uint8)
        
        # ✨ Special handling for Hue (0-180 in OpenCV)
        hue_values = center_hsv_array[:, 0]
        
        # Check if hues are bi-modal (wrap-around case)
        # This happens with red skin tones that span 0 and 180
        hue_low_count = np.sum(hue_values < 30)   # Low hue (red)
        hue_high_count = np.sum(hue_values > 150)  # High hue (red wrap)
        total_hue_outliers = hue_low_count + hue_high_count
        
        diagnostics['hue_low_count'] = int(hue_low_count)
        diagnostics['hue_high_count'] = int(hue_high_count)
        
        if total_hue_outliers > 0.3 * len(hue_values):
            # Likely hue wrap-around (red skin tones)
            lower_hue = np.percentile(hue_values[hue_values > 150], lower_percentile) if hue_high_count > 0 else np.percentile(hue_values, lower_percentile)
            upper_hue = np.percentile(hue_values[hue_values < 30], upper_percentile) if hue_low_count > 0 else np.percentile(hue_values, upper_percentile)
            diagnostics['hue_wrap_detected'] = True
        else:
            lower_hue = lower[0]
            upper_hue = upper[0]
            diagnostics['hue_wrap_detected'] = False
        
        # Ensure S and V have reasonable lower bounds
        lower_sat = max(lower[1], 15)  # Saturation >= 15 (rejects gray)
        lower_val = max(lower[2], 35)  # Value >= 35 (rejects dark)
        
        diagnostics['hue_range'] = f"{int(lower_hue)}-{int(upper_hue)}"
        diagnostics['sat_range'] = f"{int(lower_sat)}-{int(upper[1])}"
        diagnostics['val_range'] = f"{int(lower_val)}-{int(upper[2])}"
        
        # Final HSV range
        lower_final = np.array([int(lower_hue), int(lower_sat), int(lower_val)], dtype=np.uint8)
        upper_final = np.array([min(int(upper_hue), 180), int(upper[1]), int(upper[2])], dtype=np.uint8)
        
        # Validate final range
        range_size_hue = (upper_final[0] - lower_final[0]) if lower_final[0] <= upper_final[0] else (180 - lower_final[0] + upper_final[0])
        
        if range_size_hue > 80:
            diagnostics['warning'] = f'Hue range is very broad ({range_size_hue}°), consider recalibrating in better lighting'
        
        diagnostics['calibration_valid'] = True
        
        return lower_final, upper_final, diagnostics
