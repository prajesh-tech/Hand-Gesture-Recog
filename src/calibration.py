"""Persistence and validation for user-specific HSV calibration ranges."""

import json
import os
from typing import Sequence

import numpy as np


class CalibrationManager:
    """Handle validated save/load of the HSV calibration used by the application."""

    CONFIG_DIR = "config"
    CALIBRATION_FILE = os.path.join(CONFIG_DIR, "hsv_calibration.json")

    @staticmethod
    def ensure_config_dir_exists() -> None:
        """Create the local configuration directory when needed."""
        os.makedirs(CalibrationManager.CONFIG_DIR, exist_ok=True)

    @staticmethod
    def calibration_exists() -> bool:
        """Return whether a calibration file is present."""
        return os.path.isfile(CalibrationManager.CALIBRATION_FILE)

    @staticmethod
    def _normalise_hsv_range(lower_hsv: Sequence[int] | np.ndarray, upper_hsv: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Validate an HSV range before converting it to ``uint8``.

        Hue may wrap around OpenCV's 0/180 boundary, so a lower hue greater than
        the upper hue is valid. Saturation and value may not wrap.
        """
        try:
            lower = np.asarray(lower_hsv, dtype=np.int16).reshape(-1)
            upper = np.asarray(upper_hsv, dtype=np.int16).reshape(-1)
        except (TypeError, ValueError):
            return None
        if lower.shape != (3,) or upper.shape != (3,):
            return None
        if not (0 <= lower[0] <= 180 and 0 <= upper[0] <= 180):
            return None
        if not (20 <= lower[1] <= 255 and 0 <= upper[1] <= 255):
            return None
        if not (30 <= lower[2] <= 255 and 0 <= upper[2] <= 255):
            return None
        if lower[1] > upper[1] or lower[2] > upper[2]:
            return None
        return lower.astype(np.uint8), upper.astype(np.uint8)

    @staticmethod
    def save_calibration(lower_hsv: np.ndarray, upper_hsv: np.ndarray, skin_model: dict | None = None) -> bool:
        """Save a validated HSV range, returning ``False`` on a recoverable error."""
        normalised = CalibrationManager._normalise_hsv_range(lower_hsv, upper_hsv)
        if normalised is None:
            print("Calibration was not saved: invalid HSV range.")
            return False
        lower, upper = normalised
        try:
            CalibrationManager.ensure_config_dir_exists()
            data = {"lower_hsv": lower.tolist(), "upper_hsv": upper.tolist()}
            if skin_model is not None:
                data["skin_model"] = skin_model
            with open(CalibrationManager.CALIBRATION_FILE, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)
            print(f"Calibration saved to {CalibrationManager.CALIBRATION_FILE}")
            return True
        except (OSError, TypeError, ValueError) as error:
            print(f"Failed to save calibration: {error}")
            return False

    @staticmethod
    def load_calibration() -> tuple[np.ndarray, np.ndarray] | None:
        """Load a valid range or return ``None`` so the caller can recalibrate."""
        if not CalibrationManager.calibration_exists():
            return None
        try:
            with open(CalibrationManager.CALIBRATION_FILE, encoding="utf-8") as file:
                data = json.load(file)
            normalised = CalibrationManager._normalise_hsv_range(data["lower_hsv"], data["upper_hsv"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            print(f"Failed to load calibration: {error}")
            return None
        if normalised is None:
            print("Saved calibration is invalid; recalibration is required.")
        return normalised

    @staticmethod
    def load_skin_model() -> dict | None:
        """Load a persisted statistical skin model, or require recalibration."""
        if not CalibrationManager.calibration_exists():
            return None
        try:
            with open(CalibrationManager.CALIBRATION_FILE, encoding="utf-8") as file:
                model = json.load(file)["skin_model"]
            center = np.asarray(model["center"], dtype=float).reshape(-1)
            scale = np.asarray(model["scale"], dtype=float).reshape(-1)
            if center.shape != (3,) or scale.shape != (3,) or np.any(scale <= 0):
                return None
            return {"center": center.tolist(), "scale": scale.tolist()}
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def delete_calibration() -> bool:
        """Delete the calibration file, forcing calibration on the next start."""
        try:
            if CalibrationManager.calibration_exists():
                os.remove(CalibrationManager.CALIBRATION_FILE)
                print(f"Calibration deleted: {CalibrationManager.CALIBRATION_FILE}")
            return True
        except OSError as error:
            print(f"Failed to delete calibration: {error}")
            return False
