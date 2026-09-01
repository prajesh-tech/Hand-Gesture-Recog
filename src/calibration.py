"""
Calibration utilities for saving and loading user's HSV thresholds and skin models.
Persists calibration data to config/hsv_calibration.json.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np


class CalibrationManager:
    """Handle save/load of user's HSV calibration data and optional skin model."""

    CONFIG_DIR = "config"
    CALIBRATION_FILE = os.path.join(CONFIG_DIR, "hsv_calibration.json")

    @staticmethod
    def ensure_config_dir_exists() -> None:
        """Create config directory if it doesn't exist."""
        os.makedirs(CalibrationManager.CONFIG_DIR, exist_ok=True)

    @staticmethod
    def calibration_exists() -> bool:
        """Check if calibration file exists."""
        return os.path.exists(CalibrationManager.CALIBRATION_FILE)

    @staticmethod
    def save_calibration(
        lower_hsv: np.ndarray,
        upper_hsv: np.ndarray,
        statistical_model: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save HSV thresholds and optional skin model to calibration file.
        """
        try:
            CalibrationManager.ensure_config_dir_exists()

            lower_list = [int(v) for v in lower_hsv]
            upper_list = [int(v) for v in upper_hsv]

            if not CalibrationManager._is_valid_hsv_range(np.array(lower_list), np.array(upper_list)):
                print("✗ Refusing to save invalid calibration range.")
                return False

            data = {
                "lower_hsv": lower_list,
                "upper_hsv": upper_list,
                "statistical_model": statistical_model,
            }

            with open(CalibrationManager.CALIBRATION_FILE, "w") as f:
                json.dump(data, f, indent=2)

            print(f"✓ Calibration saved to {CalibrationManager.CALIBRATION_FILE}")
            return True

        except Exception as e:
            print(f"✗ Failed to save calibration: {e!s}")
            return False

    @staticmethod
    def load_calibration() -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load HSV thresholds from calibration file.

        Returns:
            Tuple of (lower_hsv, upper_hsv) or None if missing/invalid
        """
        if not CalibrationManager.calibration_exists():
            return None

        try:
            with open(CalibrationManager.CALIBRATION_FILE, "r") as f:
                data = json.load(f)

            lower = np.array(data["lower_hsv"], dtype=np.uint8)
            upper = np.array(data["upper_hsv"], dtype=np.uint8)

            if not CalibrationManager._is_valid_hsv_range(lower, upper):
                print("⚠ Loaded calibration is INVALID:")
                print(f"  Lower: {list(lower)}")
                print(f"  Upper: {list(upper)}")
                return None

            return lower, upper

        except Exception as e:
            print(f"✗ Failed to load calibration: {e!s}")
            return None

    @staticmethod
    def load_skin_model() -> Optional[Dict[str, Any]]:
        """Load statistical skin model if present in calibration file."""
        if not CalibrationManager.calibration_exists():
            return None

        try:
            with open(CalibrationManager.CALIBRATION_FILE, "r") as f:
                data = json.load(f)
            return data.get("statistical_model")
        except Exception:
            return None

    @staticmethod
    def _is_valid_hsv_range(lower: np.ndarray, upper: np.ndarray) -> bool:
        """
        Validate HSV range for OpenCV conventions.
        """
        if len(lower) != 3 or len(upper) != 3:
            return False

        low_h, low_s, low_v = int(lower[0]), int(lower[1]), int(lower[2])
        up_h, up_s, up_v = int(upper[0]), int(upper[1]), int(upper[2])

        # Hue bounds: 0-180 in OpenCV
        if low_h > 180 or up_h > 180:
            return False

        # Saturation & Value bounds: 0-255
        if low_s > 255 or up_s > 255 or low_v > 255 or up_v > 255:
            return False

        # Saturation lower bound check: S >= 15 (reject background-prone low saturation ranges)
        if low_s < 15:
            return False

        # Value lower bound check: V >= 30
        if low_v < 30:
            return False

        # S and V lower should generally be <= upper
        if low_s > up_s + 20 or low_v > up_v + 20:
            return False

        return True

    @staticmethod
    def delete_calibration() -> bool:
        """Delete calibration file."""
        try:
            if os.path.exists(CalibrationManager.CALIBRATION_FILE):
                os.remove(CalibrationManager.CALIBRATION_FILE)
                print(f"✓ Calibration deleted: {CalibrationManager.CALIBRATION_FILE}")
            return True
        except Exception as e:
            print(f"✗ Failed to delete calibration: {e!s}")
            return False