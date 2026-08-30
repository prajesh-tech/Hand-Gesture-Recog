"""
Calibration utilities for saving and loading user's HSV thresholds.
Persists calibration data to config/hsv_calibration.json.
"""

import json
import os

import numpy as np


class CalibrationManager:
    """Handle save/load of user's HSV calibration data."""
    
    CONFIG_DIR = "config"
    CALIBRATION_FILE = os.path.join(CONFIG_DIR, "hsv_calibration.json")
    
    @staticmethod
    def ensure_config_dir_exists() -> None:
        """Create config directory if it doesn't exist."""
        os.makedirs(CalibrationManager.CONFIG_DIR, exist_ok=True)
    
    @staticmethod
    def calibration_exists() -> bool:
        """Check if calibration data has been saved."""
        return os.path.exists(CalibrationManager.CALIBRATION_FILE)
    
    @staticmethod
    def save_calibration(lower_hsv: np.ndarray, upper_hsv: np.ndarray) -> bool:
        """
        Save HSV thresholds to calibration file.
        
        Args:
            lower_hsv: Lower HSV threshold (3-element array)
            upper_hsv: Upper HSV threshold (3-element array)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            CalibrationManager.ensure_config_dir_exists()
            
            data = {
                'lower_hsv': lower_hsv.tolist(),
                'upper_hsv': upper_hsv.tolist()
            }
            
            with open(CalibrationManager.CALIBRATION_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Calibration saved to {CalibrationManager.CALIBRATION_FILE}")
            return True
        
        except Exception as e:  # noqa: BLE001
            print(f"✗ Failed to save calibration: {e!s}")
            return False
    
    @staticmethod
    def load_calibration() -> tuple[np.ndarray, np.ndarray] | None:
        """
        Load HSV thresholds from calibration file.
        
        Returns:
            Tuple of (lower_hsv, upper_hsv) or None if file doesn't exist or is invalid
        """
        if not CalibrationManager.calibration_exists():
            return None
        
        try:
            with open(CalibrationManager.CALIBRATION_FILE, 'r') as f:
                data = json.load(f)
            
            lower = np.array(data['lower_hsv'], dtype=np.uint8)
            upper = np.array(data['upper_hsv'], dtype=np.uint8)
            
            # Validate HSV ranges
            if not CalibrationManager._is_valid_hsv_range(lower, upper):
                print("⚠ Loaded calibration is INVALID:")
                print(f"  Lower: {list(lower)}")
                print(f"  Upper: {list(upper)}")
                print("  HSV Hue must be 0-180 in OpenCV!")
                print("  Returning None to force recalibration...")
                return None
            
            return lower, upper
        
        except Exception as e:  # noqa: BLE001
            print(f"✗ Failed to load calibration: {e!s}")
            return None
    
    @staticmethod
    def _is_valid_hsv_range(lower: np.ndarray, upper: np.ndarray) -> bool:
        """
        Validate HSV range for OpenCV conventions.
        
        Args:
            lower: Lower HSV threshold [H, S, V]
            upper: Upper HSV threshold [H, S, V]
        
        Returns:
            True if range is valid, False otherwise
        """
        # Hue channel: 0-180 in OpenCV
        if lower[0] > 180 or upper[0] > 180:
            return False
        
        # Saturation and Value: 0-255
        if any(v > 255 for v in lower) or any(v > 255 for v in upper):
            return False
        
        # Lower should generally be <= upper (some tolerance for wrapping hue)
        if lower[1] > upper[1] + 20 or lower[2] > upper[2] + 20:  # S and V  # noqa: SIM103
            return False
        
        return True
    
    @staticmethod
    def delete_calibration() -> bool:
        """Delete calibration file (forces re-calibration on next run)."""
        try:
            if os.path.exists(CalibrationManager.CALIBRATION_FILE):
                os.remove(CalibrationManager.CALIBRATION_FILE)
                print(f"✓ Calibration deleted: {CalibrationManager.CALIBRATION_FILE}")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"✗ Failed to delete calibration: {e!s}")
            return False