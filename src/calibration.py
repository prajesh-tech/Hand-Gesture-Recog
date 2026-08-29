"""
Calibration utilities for saving and loading user's HSV thresholds.
Persists calibration data to config/hsv_calibration.json.
"""

import json
import os
import numpy as np
from typing import Tuple, Optional


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
        
        except Exception as e:
            print(f"✗ Failed to save calibration: {str(e)}")
            return False
    
    @staticmethod
    def load_calibration() -> Optional[Tuple[np.ndarray, np.ndarray]]:
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
            
            return lower, upper
        
        except Exception as e:
            print(f"✗ Failed to load calibration: {str(e)}")
            return None
    
    @staticmethod
    def delete_calibration() -> bool:
        """Delete calibration file (forces re-calibration on next run)."""
        try:
            if os.path.exists(CalibrationManager.CALIBRATION_FILE):
                os.remove(CalibrationManager.CALIBRATION_FILE)
                print(f"✓ Calibration deleted: {CalibrationManager.CALIBRATION_FILE}")
            return True
        except Exception as e:
            print(f"✗ Failed to delete calibration: {str(e)}")
            return False
