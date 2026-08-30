"""Explainable, classical contour-feature gesture recognition."""

from typing import Any

import cv2
import numpy as np


class GestureRecognizer:
    """Recognize four gestures from a validated hand contour.

    Thresholds are deliberately exposed through the constructor and update methods;
    they are starting points to tune from ``main_debug.py`` observations.
    """

    THRESHOLDS = {
        "min_hand_area": 500.0,
        "min_defect_depth": 12.0,
        "palm_defects_min": 3,
        "palm_solidity_max": 0.88,
        "two_fingers_defects_min": 1,
        "two_fingers_defects_max": 2,
        "two_fingers_elongation_min": 1.25,
        "one_finger_defects_max": 1,
        "one_finger_elongation_min": 2.0,
        "fist_solidity_min": 0.88,
        "fist_extent_min": 0.60,
        "fist_elongation_max": 1.45,
    }

    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        self.thresholds = self.THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)

    def extract_features(self, contour: np.ndarray) -> dict[str, Any] | None:
        """Return stable geometric measurements, or ``None`` for an invalid contour."""
        if not isinstance(contour, np.ndarray) or len(contour) < 3:
            return None
        try:
            area = float(cv2.contourArea(contour))
            if area < self.thresholds["min_hand_area"]:
                return None
            perimeter = float(cv2.arcLength(contour, closed=True))
            x, y, width, height = cv2.boundingRect(contour)
            if width <= 0 or height <= 0 or perimeter <= 0:
                return None
            box_area = float(width * height)
            hull_points = cv2.convexHull(contour, returnPoints=True)
            hull_area = float(cv2.contourArea(hull_points))
            if hull_area <= 0:
                return None
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            defects_count = 0
            if hull_indices is not None and len(hull_indices) >= 3:
                defects = cv2.convexityDefects(contour, hull_indices)
                if defects is not None:
                    defects_count = sum(
                        1 for defect in defects.reshape(-1, 4)
                        if defect[3] / 256.0 >= self.thresholds["min_defect_depth"]
                    )
            aspect_ratio = width / height
            return {
                "area": area,
                "perimeter": perimeter,
                "bbox": (int(x), int(y), int(width), int(height)),
                "aspect_ratio": float(aspect_ratio),
                "elongation": float(max(aspect_ratio, 1.0 / aspect_ratio)),
                "extent": area / box_area,
                "solidity": area / hull_area,
                "perimeter_area_ratio": perimeter / area,
                "circularity": (4.0 * np.pi * area) / (perimeter * perimeter),
                "convexity_defects_count": int(defects_count),
                "hull": hull_points,
                "hull_indices": hull_indices,
                "hull_area": hull_area,
            }
        except cv2.error:
            return None

    def recognize_gesture(self, contour: np.ndarray) -> str | None:
        """Return a gesture label, ``Unknown``, or ``None`` when no hand is valid."""
        features = self.extract_features(contour)
        return None if features is None else self._classify_by_features(features)

    def _classify_by_features(self, features: dict[str, Any]) -> str:
        """Classify a valid feature set in distinctive-to-compact order."""
        defects = features["convexity_defects_count"]
        solidity = features["solidity"]
        extent = features["extent"]
        elongation = features["elongation"]
        if defects >= self.thresholds["palm_defects_min"] and solidity <= self.thresholds["palm_solidity_max"]:
            return "Open Palm"
        if (self.thresholds["two_fingers_defects_min"] <= defects <= self.thresholds["two_fingers_defects_max"]
                and elongation >= self.thresholds["two_fingers_elongation_min"]):
            return "Two Fingers"
        if defects <= self.thresholds["one_finger_defects_max"] and elongation >= self.thresholds["one_finger_elongation_min"]:
            return "One Finger"
        if (defects == 0 and solidity >= self.thresholds["fist_solidity_min"]
                and extent >= self.thresholds["fist_extent_min"]
                and elongation <= self.thresholds["fist_elongation_max"]):
            return "Fist"
        return "Unknown"

    def set_threshold(self, key: str, value: float) -> None:
        if key not in self.thresholds:
            raise KeyError(f"Unknown gesture threshold: {key}")
        self.thresholds[key] = value

    def update_thresholds(self, thresholds: dict[str, float]) -> None:
        for key, value in thresholds.items():
            self.set_threshold(key, value)

    def get_threshold(self, key: str) -> float | None:
        return self.thresholds.get(key)

    def get_all_thresholds(self) -> dict[str, float]:
        return self.thresholds.copy()
