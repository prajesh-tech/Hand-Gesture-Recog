"""Tests for persisted HSV calibration validation."""

import json

import numpy as np

from src.calibration import CalibrationManager


def test_load_corrupt_calibration_returns_none(tmp_path, monkeypatch):
    path = tmp_path / "hsv_calibration.json"
    path.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(CalibrationManager, "CALIBRATION_FILE", str(path))
    assert CalibrationManager.load_calibration() is None


def test_load_rejects_out_of_range_values_before_uint8_wrap(tmp_path, monkeypatch):
    path = tmp_path / "hsv_calibration.json"
    path.write_text(json.dumps({"lower_hsv": [0, 40, 40], "upper_hsv": [300, 200, 200]}), encoding="utf-8")
    monkeypatch.setattr(CalibrationManager, "CALIBRATION_FILE", str(path))
    assert CalibrationManager.load_calibration() is None


def test_load_rejects_background_prone_low_saturation_range(tmp_path, monkeypatch):
    path = tmp_path / "hsv_calibration.json"
    path.write_text(json.dumps({"lower_hsv": [0, 7, 53], "upper_hsv": [20, 138, 191]}), encoding="utf-8")
    monkeypatch.setattr(CalibrationManager, "CALIBRATION_FILE", str(path))
    assert CalibrationManager.load_calibration() is None


def test_save_and_load_valid_hue_wrap_range(tmp_path, monkeypatch):
    path = tmp_path / "hsv_calibration.json"
    monkeypatch.setattr(CalibrationManager, "CALIBRATION_FILE", str(path))
    monkeypatch.setattr(CalibrationManager, "CONFIG_DIR", str(tmp_path))
    assert CalibrationManager.save_calibration(np.array([170, 40, 50]), np.array([10, 200, 220]))
    lower, upper = CalibrationManager.load_calibration()
    assert lower.tolist() == [170, 40, 50]
    assert upper.tolist() == [10, 200, 220]
