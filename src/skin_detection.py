"""
Skin detection module using HSV color space thresholding and statistical modeling.
Provides robust HSV skin segmentation, morphological cleanup, and percentile calibration.
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class SkinDetector:
    """HSV-based skin color detection with configurable thresholds and robust calibration."""

    # Default HSV ranges for typical skin tone
    DEFAULT_LOWER_HSV = np.array([0, 15, 40], dtype=np.uint8)
    DEFAULT_UPPER_HSV = np.array([20, 170, 255], dtype=np.uint8)

    def __init__(
        self,
        lower_hsv: Optional[np.ndarray] = None,
        upper_hsv: Optional[np.ndarray] = None,
        morphology_kernel_size: int = 5,
        blur_kernel_size: int = 0,
        morphology_strength: str = "normal",
        statistical_model: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize skin detector.

        Args:
            lower_hsv: Lower HSV threshold (3-element array [H, S, V])
            upper_hsv: Upper HSV threshold (3-element array [H, S, V])
            morphology_kernel_size: Size of morphological kernel (odd integer)
            blur_kernel_size: Optional Gaussian blur kernel size (0 to disable)
            morphology_strength: 'light', 'normal', or 'strong'
            statistical_model: Optional precomputed Gaussian/Mahalanobis skin parameters
        """
        self.lower_hsv = (
            np.array(lower_hsv, dtype=np.uint8)
            if lower_hsv is not None
            else self.DEFAULT_LOWER_HSV.copy()
        )
        self.upper_hsv = (
            np.array(upper_hsv, dtype=np.uint8)
            if upper_hsv is not None
            else self.DEFAULT_UPPER_HSV.copy()
        )

        if morphology_kernel_size % 2 == 0:
            morphology_kernel_size += 1
        self.morphology_kernel_size = morphology_kernel_size
        self.blur_kernel_size = blur_kernel_size
        self.morphology_strength = morphology_strength

        self.morphology_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morphology_kernel_size, morphology_kernel_size),
        )

        self.statistical_model = statistical_model
        self.use_statistical_model = statistical_model is not None

    def set_hsv_range(self, lower_hsv: np.ndarray, upper_hsv: np.ndarray) -> None:
        """Update HSV thresholds."""
        self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        self.use_statistical_model = False

    def get_hsv_range(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current HSV thresholds."""
        return self.lower_hsv.copy(), self.upper_hsv.copy()

    def detect_skin(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Detect skin regions in BGR frame.

        Args:
            frame_bgr: Input frame in BGR color space

        Returns:
            Binary mask where white (255) indicates detected skin, black (0) otherwise
        """
        if (
            frame_bgr is None
            or not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.size == 0
            or frame_bgr.ndim != 3
            or frame_bgr.shape[0] == 0
            or frame_bgr.shape[1] == 0
        ):
            raise ValueError("Invalid frame: frame_bgr must be a non-empty 3D BGR image")

        # Apply Gaussian blur if configured
        if self.blur_kernel_size > 0:
            ksize = self.blur_kernel_size if self.blur_kernel_size % 2 != 0 else self.blur_kernel_size + 1
            frame_proc = cv2.GaussianBlur(frame_bgr, (ksize, ksize), 0)
        else:
            frame_proc = frame_bgr

        # Convert BGR to HSV
        frame_hsv = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2HSV)

        # Apply threshold range (handling potential hue wrap)
        mask = self._apply_hsv_range(frame_hsv, self.lower_hsv, self.upper_hsv)
        return mask

    def _apply_hsv_range(
        self, frame_hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> np.ndarray:
        """Apply HSV threshold, ensuring uint8 array types and handling hue wrap-around."""
        lower_arr = np.array(lower, dtype=np.uint8)
        upper_arr = np.array(upper, dtype=np.uint8)

        # Check if hue wraps around (lower_hue > upper_hue)
        if int(lower_arr[0]) > int(upper_arr[0]):
            mask1 = cv2.inRange(
                frame_hsv,
                np.array([lower_arr[0], lower_arr[1], lower_arr[2]], dtype=np.uint8),
                np.array([180, upper_arr[1], upper_arr[2]], dtype=np.uint8),
            )
            mask2 = cv2.inRange(
                frame_hsv,
                np.array([0, lower_arr[1], lower_arr[2]], dtype=np.uint8),
                np.array([upper_arr[0], upper_arr[1], upper_arr[2]], dtype=np.uint8),
            )
            return cv2.bitwise_or(mask1, mask2)
        else:
            return cv2.inRange(frame_hsv, lower_arr, upper_arr)

    def apply_morphology(self, mask: np.ndarray, operation: str = "both") -> np.ndarray:
        """
        Apply morphological operations to clean up binary mask.

        Args:
            mask: Binary mask
            operation: 'open', 'close', 'both', 'none', or None

        Returns:
            Cleaned binary mask
        """
        valid_ops = {"open", "close", "both", "none", None}
        if operation not in valid_ops:
            raise ValueError(f"Invalid morphology operation: '{operation}'. Must be one of {valid_ops}")

        if operation == "none" or operation is None or mask is None or mask.size == 0:
            return mask

        if self.morphology_strength == "light":
            open_iter, close_iter = 1, 0
        elif self.morphology_strength == "strong":
            open_iter, close_iter = 2, 2
        else:  # normal
            open_iter, close_iter = 2, 1

        result = mask.copy()
        if operation in ("open", "both") and open_iter > 0:
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, self.morphology_kernel, iterations=open_iter)

        if operation in ("close", "both") and close_iter > 0:
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, self.morphology_kernel, iterations=close_iter)

        return result

    def detect_and_clean(self, frame_bgr: np.ndarray, morphology_op: str = "both") -> np.ndarray:
        """Detect skin and apply morphological cleanup in one step."""
        mask = self.detect_skin(frame_bgr)
        return self.apply_morphology(mask, operation=morphology_op)

    @staticmethod
    def valid_calibration_pixel_count(
        sample_region: np.ndarray, roi_mask: Optional[np.ndarray] = None
    ) -> int:
        """
        Count non-dark, non-desaturated skin-candidate pixels in ROI (S > 20 and V > 40).
        """
        if sample_region is None or sample_region.size == 0 or sample_region.ndim != 3:
            return 0

        hsv = cv2.cvtColor(sample_region, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        valid_condition = (sat > 20) & (val > 40)
        if roi_mask is not None and roi_mask.shape[:2] == sample_region.shape[:2]:
            valid_condition &= roi_mask > 0

        return int(np.count_nonzero(valid_condition))

    @staticmethod
    def extract_hsv_from_region(
        frame_bgr: np.ndarray, region_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute robust percentile-based HSV bounds from a masked ROI.
        Handles hue wrap-around and prevents uint8 underflow/overflow.
        """
        if frame_bgr is None or frame_bgr.size == 0 or region_mask is None or region_mask.size == 0:
            return SkinDetector.DEFAULT_LOWER_HSV.copy(), SkinDetector.DEFAULT_UPPER_HSV.copy()

        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        pixels = frame_hsv[region_mask > 0]

        if len(pixels) == 0:
            return SkinDetector.DEFAULT_LOWER_HSV.copy(), SkinDetector.DEFAULT_UPPER_HSV.copy()

        # Filter out dark / desaturated background pixels
        valid_mask = (pixels[:, 1] > 20) & (pixels[:, 2] > 40)
        valid_pixels = pixels[valid_mask]

        if len(valid_pixels) == 0:
            valid_pixels = pixels  # Fallback if all filtered

        # Percentile range (5th - 95th)
        lower_raw = np.percentile(valid_pixels, 5, axis=0)
        upper_raw = np.percentile(valid_pixels, 95, axis=0)

        # Convert numpy scalars to Python int to prevent uint8 underflow
        low_h = int(lower_raw[0])
        low_s = int(lower_raw[1])
        low_v = int(lower_raw[2])

        up_h = int(upper_raw[0])
        up_s = int(upper_raw[1])
        up_v = int(upper_raw[2])

        # Check for hue wrap-around (red skin tones near 0 and 180)
        hues = valid_pixels[:, 0]
        hue_low_cnt = np.sum(hues < 30)
        hue_high_cnt = np.sum(hues > 150)
        if (hue_low_cnt + hue_high_cnt) > 0.3 * len(hues) and hue_low_cnt > 0 and hue_high_cnt > 0:
            # Wrap detected
            high_hues = hues[hues > 150]
            low_hues = hues[hues < 30]
            final_low_h = int(np.percentile(high_hues, 5)) if len(high_hues) > 0 else low_h
            final_up_h = int(np.percentile(low_hues, 95)) if len(low_hues) > 0 else up_h
        else:
            final_low_h = max(0, low_h - 10)
            final_up_h = min(180, up_h + 10)

        # Enforce minimum S >= 15 and V >= 35 bounds to reject dark/desaturated background
        final_low_s = max(15, max(0, low_s - 10))
        final_low_v = max(35, max(0, low_v - 10))

        final_up_s = min(255, up_s + 15)
        final_up_v = min(255, up_v + 15)

        lower_bound = np.array([final_low_h, final_low_s, final_low_v], dtype=np.uint8)
        upper_bound = np.array([final_up_h, final_up_s, final_up_v], dtype=np.uint8)

        return lower_bound, upper_bound

    @staticmethod
    def calibrate_from_samples(
        sample_frames: list, morph_kernel_size: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Calibrate HSV thresholds from multiple BGR sample frames using center ROI.
        """
        if not sample_frames:
            return (
                SkinDetector.DEFAULT_LOWER_HSV.copy(),
                SkinDetector.DEFAULT_UPPER_HSV.copy(),
                {"calibration_valid": False, "error": "No sample frames provided"},
            )

        valid_pixels_list = []
        total_roi_pixels = 0

        for frame in sample_frames:
            h, w = frame.shape[:2]
            cy, cx = h // 2, w // 2
            # Extract inner 50% central region
            rh, rw = int(h * 0.25), int(w * 0.25)
            y1, y2 = max(0, cy - rh), min(h, cy + rh)
            x1, x2 = max(0, cx - rw), min(w, cx + rw)

            roi = frame[y1:y2, x1:x2]
            total_roi_pixels += roi.shape[0] * roi.shape[1]

            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            # Filter S > 20 and V > 40
            mask = (roi_hsv[:, 1] > 20) & (roi_hsv[:, 2] > 40)
            valid_pixels_list.extend(roi_hsv[mask])

        diagnostics: Dict[str, Any] = {
            "total_samples": len(sample_frames),
            "total_roi_pixels": total_roi_pixels,
            "valid_pixels": len(valid_pixels_list),
            "valid_percentage": (100.0 * len(valid_pixels_list) / total_roi_pixels) if total_roi_pixels > 0 else 0.0,
        }

        if len(valid_pixels_list) < 50:
            diagnostics["calibration_valid"] = False
            diagnostics["error"] = "Insufficient valid skin pixels in samples"
            return SkinDetector.DEFAULT_LOWER_HSV.copy(), SkinDetector.DEFAULT_UPPER_HSV.copy(), diagnostics

        arr = np.array(valid_pixels_list, dtype=np.float32)

        # Check hue wrap
        hues = arr[:, 0]
        hue_low_cnt = np.sum(hues < 30)
        hue_high_cnt = np.sum(hues > 150)
        wrap_detected = (hue_low_cnt + hue_high_cnt) > 0.3 * len(hues) and hue_low_cnt > 0 and hue_high_cnt > 0
        diagnostics["hue_wrap_detected"] = wrap_detected
        diagnostics["hue_low_count"] = int(hue_low_cnt)
        diagnostics["hue_high_count"] = int(hue_high_cnt)

        if wrap_detected:
            high_h = hues[hues > 150]
            low_h = hues[hues < 30]
            l_h = int(np.percentile(high_h, 5)) if len(high_h) > 0 else int(np.percentile(hues, 5))
            u_h = int(np.percentile(low_h, 95)) if len(low_h) > 0 else int(np.percentile(hues, 95))
        else:
            l_h = max(0, int(np.percentile(hues, 5)) - 10)
            u_h = min(180, int(np.percentile(hues, 95)) + 10)

        sats = arr[:, 1]
        vals = arr[:, 2]

        l_s = max(15, max(0, int(np.percentile(sats, 5)) - 10))
        u_s = min(255, int(np.percentile(sats, 95)) + 15)

        l_v = max(35, max(0, int(np.percentile(vals, 5)) - 10))
        u_v = min(255, int(np.percentile(vals, 95)) + 15)

        lower = np.array([l_h, l_s, l_v], dtype=np.uint8)
        upper = np.array([u_h, u_s, u_v], dtype=np.uint8)

        if not SkinDetector.is_valid_hsv_range(lower, upper):
            diagnostics["calibration_valid"] = False
            diagnostics["error"] = "Calibration result produced invalid or excessively broad HSV bounds"
            return SkinDetector.DEFAULT_LOWER_HSV.copy(), SkinDetector.DEFAULT_UPPER_HSV.copy(), diagnostics

        diagnostics["hue_range"] = f"{l_h}-{u_h}"
        diagnostics["sat_range"] = f"{l_s}-{u_s}"
        diagnostics["val_range"] = f"{l_v}-{u_v}"
        diagnostics["calibration_valid"] = True

        return lower, upper, diagnostics

    @staticmethod
    def is_valid_hsv_range(lower: np.ndarray, upper: np.ndarray) -> bool:
        """
        Validate HSV range for OpenCV conventions (H: 0-180, S: 0-255, V: 0-255).
        Supports both normal ranges (lower_h <= upper_h) and wrap-around ranges (lower_h > upper_h).
        """
        if lower is None or upper is None or len(lower) != 3 or len(upper) != 3:
            return False

        low_h, low_s, low_v = int(lower[0]), int(lower[1]), int(lower[2])
        up_h, up_s, up_v = int(upper[0]), int(upper[1]), int(upper[2])

        # Hue bounds: 0-180 in OpenCV
        if low_h < 0 or low_h > 180 or up_h < 0 or up_h > 180:
            return False

        # Saturation & Value bounds: 0-255
        if low_s < 0 or low_s > 255 or up_s < 0 or up_s > 255 or low_v < 0 or low_v > 255 or up_v < 0 or up_v > 255:
            return False

        # Saturation lower bound check: S >= 15 (reject background-prone low saturation ranges)
        if low_s < 15:
            return False

        # Value lower bound check: V >= 30
        if low_v < 30:
            return False

        # Saturation and Value lower bound must be <= upper bound
        if low_s > up_s or low_v > up_v:
            return False

        # Hue range check: normal vs wrap-around
        if low_h <= up_h:
            hue_span = up_h - low_h
        else:
            # Wrap-around range: e.g. [162, 15, 40] to [16, 170, 255]
            # Requires low_h in upper hue range (>= 120) and up_h in lower hue range (<= 60)
            if low_h < 120 or up_h > 60:
                return False
            hue_span = (180 - low_h) + up_h

        # Reject excessively broad hue ranges (e.g. > 110 degrees out of 180)
        if hue_span > 110:
            return False

        return True


    @staticmethod
    def build_statistical_model(
        sample_bgr: np.ndarray, roi_mask: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Build Gaussian mean & covariance statistical model from skin pixels.
        """
        if sample_bgr is None or sample_bgr.size == 0:
            return None

        hsv = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        if roi_mask is not None:
            mask_flat = (roi_mask.reshape(-1) > 0) & (pixels[:, 1] > 20) & (pixels[:, 2] > 40)
            pixels = pixels[mask_flat]

        if len(pixels) < 20:
            return None

        mean = np.mean(pixels, axis=0)
        cov = np.cov(pixels, rowvar=False)

        return {"mean": mean.tolist(), "cov": cov.tolist()}

    def detect_skin_statistical(
        self, frame_bgr: np.ndarray, distance_threshold: float = 2.5
    ) -> np.ndarray:
        """
        Detect skin using Mahalanobis distance from statistical model.
        """
        if not self.use_statistical_model or self.statistical_model is None:
            return self.detect_skin(frame_bgr)

        try:
            mean = np.array(self.statistical_model["mean"], dtype=np.float32)
            cov = np.array(self.statistical_model["cov"], dtype=np.float32) + np.eye(3) * 0.1
            cov_inv = np.linalg.inv(cov)

            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            h, w = hsv.shape[:2]
            pixels = hsv.reshape(-1, 3).astype(np.float32)

            diff = pixels - mean
            dist = np.sqrt(np.sum((diff @ cov_inv) * diff, axis=1))

            mask = (dist < distance_threshold).astype(np.uint8) * 255
            return mask.reshape(h, w)
        except Exception:
            return self.detect_skin(frame_bgr)
