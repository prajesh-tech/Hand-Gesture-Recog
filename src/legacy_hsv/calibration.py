"""
Calibration utilities for saving and loading user's HSV thresholds.
Persists calibration data to project-relative config/hsv_calibration.json.
"""

import json
import os
from typing import Optional, Tuple

import numpy as np

from src.results import CalibrationResult


class CalibrationManager:
    """Handle save/load of user's HSV calibration data."""

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
    CALIBRATION_FILE = os.path.join(CONFIG_DIR, "hsv_calibration.json")

    @staticmethod
    def get_calibration_path() -> str:
        """Get project-relative calibration file path."""
        return CalibrationManager.CALIBRATION_FILE

    @staticmethod
    def ensure_config_dir_exists() -> None:
        """Create config directory if it doesn't exist."""
        config_dir = os.path.dirname(CalibrationManager.CALIBRATION_FILE)
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)

    @staticmethod
    def calibration_exists() -> bool:
        """Check if calibration file exists."""
        return os.path.exists(CalibrationManager.CALIBRATION_FILE)

    @staticmethod
    def save_calibration(
        lower_hsv: np.ndarray,
        upper_hsv: np.ndarray,
    ) -> bool:
        """
        Save HSV thresholds to calibration file.
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

            if "lower_hsv" not in data or "upper_hsv" not in data:
                return None

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
    def load_calibration_result() -> CalibrationResult:
        """Load calibration into structured CalibrationResult."""
        loaded = CalibrationManager.load_calibration()
        if loaded is None:
            return CalibrationResult(
                success=False,
                status="FAILED",
                message="No valid calibration file found",
            )
        lower, upper = loaded
        return CalibrationResult(
            success=True,
            status="SUCCESS",
            lower_hsv=lower,
            upper_hsv=upper,
            message="Calibration loaded successfully",
        )

    @staticmethod
    def _is_valid_hsv_range(lower: np.ndarray, upper: np.ndarray) -> bool:
        """Validate HSV range using SkinDetector.is_valid_hsv_range."""
        from src.legacy_hsv.skin_detection import SkinDetector
        return SkinDetector.is_valid_hsv_range(lower, upper)

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