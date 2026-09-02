"""
Tests for HSV calibration validation, hue wrap-around, persistence, and recalibration safety.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.calibration import CalibrationManager
from src.skin_detection import SkinDetector
from src.results import CalibrationResult


def test_save_and_load_valid_normal_calibration(tmp_path, monkeypatch):
    """Test saving and loading a normal valid HSV calibration range."""
    path = tmp_path / "hsv_calibration.json"
    monkeypatch.setattr(CalibrationManager, "CALIBRATION_FILE", str(path))
    monkeypatch.setattr(CalibrationManager, "CONFIG_DIR", str(tmp_path))

    lower = np.array([20, 40, 50], dtype=np.uint8)
    upper = np.array([80, 200, 220], dtype=np.uint8)

    assert CalibrationManager.save_calibration(lower, upper) is True
    loaded = CalibrationManager.load_calibration()
    assert loaded is not None
    loaded_lower, loaded_upper = loaded
    assert loaded_lower.tolist() == [20, 40, 50]
    assert loaded_upper.tolist() == [80, 200, 220]


def test_save_and_load_valid_hue_wrap_calibration(tmp_path, monkeypatch):
    """Test saving and loading a valid hue wrap-around calibration range [162, 16]."""
    path = tmp_path / "hsv_calibration.json"
    monkeypatch.setattr(CalibrationManager, "CALIBRATION_FILE", str(path))
    monkeypatch.setattr(CalibrationManager, "CONFIG_DIR", str(tmp_path))

    lower = np.array([162, 40, 50], dtype=np.uint8)
    upper = np.array([16, 200, 220], dtype=np.uint8)

    assert CalibrationManager.save_calibration(lower, upper) is True
    loaded = CalibrationManager.load_calibration()
    assert loaded is not None
    loaded_lower, loaded_upper = loaded
    assert loaded_lower.tolist() == [162, 40, 50]
    assert loaded_upper.tolist() == [16, 200, 220]


def test_load_corrupt_calibration_returns_none(tmp_path, monkeypatch):
    """Test that corrupt JSON files are rejected on load."""
    path = tmp_path / "hsv_calibration.json"
    path.write_text("not valid json", encoding="utf-8")
    monkeypatch.setattr(CalibrationManager, "CALIBRATION_FILE", str(path))

    assert CalibrationManager.load_calibration() is None
    res = CalibrationManager.load_calibration_result()
    assert res.success is False
    assert res.status == "FAILED"


def test_rejects_out_of_range_values(tmp_path, monkeypatch):
    """Test rejection of out-of-bounds H, S, or V values."""
    path = tmp_path / "hsv_calibration.json"
    path.write_text(json.dumps({"lower_hsv": [0, 40, 40], "upper_hsv": [300, 200, 200]}), encoding="utf-8")
    monkeypatch.setattr(CalibrationManager, "CALIBRATION_FILE", str(path))

    assert CalibrationManager.load_calibration() is None


def test_rejects_low_saturation_or_value_range(tmp_path, monkeypatch):
    """Test rejection of background-prone low saturation (S < 15) or low value (V < 30)."""
    path = tmp_path / "hsv_calibration.json"
    path.write_text(json.dumps({"lower_hsv": [0, 7, 53], "upper_hsv": [20, 138, 191]}), encoding="utf-8")
    monkeypatch.setattr(CalibrationManager, "CALIBRATION_FILE", str(path))

    assert CalibrationManager.load_calibration() is None


def test_rejects_invalid_s_or_v_ordering():
    """Test rejection when lower S/V is greater than upper S/V."""
    lower = np.array([10, 150, 100], dtype=np.uint8)
    upper = np.array([30, 100, 200], dtype=np.uint8)

    assert SkinDetector.is_valid_hsv_range(lower, upper) is False


def test_rejects_excessively_broad_hue_range():
    """Test rejection when hue span exceeds 110 degrees."""
    # Normal range with span = 150 - 10 = 140 (> 110)
    broad_normal_lower = np.array([10, 30, 40], dtype=np.uint8)
    broad_normal_upper = np.array([150, 200, 220], dtype=np.uint8)
    assert SkinDetector.is_valid_hsv_range(broad_normal_lower, broad_normal_upper) is False

    # Backwards invalid wrap-around range [60, 50] (span = (180-60) + 50 = 170)
    broad_wrap_lower = np.array([60, 30, 40], dtype=np.uint8)
    broad_wrap_upper = np.array([50, 200, 220], dtype=np.uint8)
    assert SkinDetector.is_valid_hsv_range(broad_wrap_lower, broad_wrap_upper) is False


def test_calibration_file_path_is_relative_to_project():
    """Test that CalibrationManager resolves path relative to project root, independent of CWD."""
    path = CalibrationManager.get_calibration_path()
    assert "config" in path
    assert "hsv_calibration.json" in path
    assert os.path.isabs(path)


def test_calibrate_from_samples_insufficient_pixels():
    """Test calibration failure when sample frames have fewer than 50 valid pixels."""
    black_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    lower, upper, stats = SkinDetector.calibrate_from_samples([black_frame])
    assert stats["calibration_valid"] is False
    assert "Insufficient valid skin pixels" in stats["error"]
