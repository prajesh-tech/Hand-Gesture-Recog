"""Explainable, classical contour-feature gesture recognition with heuristic confidence score."""

from typing import Any, Dict, Optional
import cv2
import numpy as np

from src.results import GestureResult


class GestureRecognizer:
    """Recognize four gestures from a validated hand contour.

    Thresholds are deliberately exposed through the constructor and update methods;
    they are starting points to tune from ``main_debug.py`` observations.
    DO NOT modify existing classification thresholds.
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

    def __init__(self, thresholds: Optional[Dict[str, float]] = None) -> None:
        self.thresholds = self.THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)

    def extract_features(self, contour: np.ndarray) -> Optional[Dict[str, Any]]:
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
                        1
                        for defect in defects.reshape(-1, 4)
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

    def recognize_gesture(self, contour: np.ndarray) -> Optional[str]:
        """Return a gesture label, ``Unknown``, or ``None`` when no hand is valid."""
        features = self.extract_features(contour)
        return None if features is None else self._classify_by_features(features)

    def _classify_by_features(self, features: Dict[str, Any]) -> str:
        """Classify a valid feature set: compact/specific gestures first, open palm last.

        Evaluation order (most-constrained to least-constrained):
        1. Fist        — zero defects + high solidity/extent, compact shape
        2. One Finger  — ≤1 defect + strong elongation (narrow pointing shape)
        3. Two Fingers — 1-2 defects + moderate elongation (forked, elongated shape)
        4. Open Palm   — 3+ defects + low solidity + non-elongated (spread hand)
        5. Unknown     — anything else

        Open Palm is checked *last* among the named gestures so that elongated
        finger gestures with noisy extra defects cannot fall into it.
        """
        defects = features["convexity_defects_count"]
        solidity = features["solidity"]
        extent = features["extent"]
        elongation = features["elongation"]

        # 1. Fist — most compact, zero convexity defects
        if (
            defects == 0
            and solidity >= self.thresholds["fist_solidity_min"]
            and extent >= self.thresholds["fist_extent_min"]
            and elongation <= self.thresholds["fist_elongation_max"]
        ):
            return "Fist"

        # 2. One Finger — strongly elongated, at most 1 defect
        if defects <= self.thresholds["one_finger_defects_max"] and elongation >= self.thresholds["one_finger_elongation_min"]:
            return "One Finger"

        # 3. Two Fingers — moderately elongated fork, 1–2 defects
        if (
            self.thresholds["two_fingers_defects_min"] <= defects <= self.thresholds["two_fingers_defects_max"]
            and elongation >= self.thresholds["two_fingers_elongation_min"]
        ):
            return "Two Fingers"

        # 4. Open Palm — many defects, low solidity, and NOT highly elongated
        #    (elongation < fist_elongation_max * 1.2 keeps elongated finger shapes out)
        if (
            defects >= self.thresholds["palm_defects_min"]
            and solidity <= self.thresholds["palm_solidity_max"]
            and elongation <= self.thresholds["fist_elongation_max"] * 1.2
        ):
            return "Open Palm"

        return "Unknown"

    def calculate_confidence(
        self,
        contour: Optional[np.ndarray],
        contour_score: float = 0.0,
        features: Optional[Dict[str, Any]] = None,
        gesture_label: Optional[str] = None,
        history_confidence: float = 0.0,
    ) -> float:
        """
        Calculate heuristic detection confidence score from 0.0 to 100.0%.

        Combines:
        - Contour detection quality score
        - Shape geometric plausibility (solidity, extent, aspect ratio)
        - Gesture classification margin
        - Temporal history consistency
        """
        if contour is None or gesture_label is None:
            return 0.0

        # 1. Contour score contribution (0 - 35 pts)
        score_contrib = min(35.0, max(5.0, contour_score * 5.0))

        # 2. Geometric plausibility contribution (0 - 30 pts)
        geom_contrib = 0.0
        if features is not None:
            solidity = features.get("solidity", 0.0)
            extent = features.get("extent", 0.0)
            aspect_ratio = features.get("aspect_ratio", 1.0)

            if 0.4 <= solidity <= 0.95:
                geom_contrib += 10.0
            if 0.25 <= extent <= 0.85:
                geom_contrib += 10.0
            if 0.3 <= aspect_ratio <= 3.5:
                geom_contrib += 10.0

        # 3. Gesture validity contribution (0 - 25 pts). This fixed bonus
        # reflects a label matching one of the supported gesture definitions;
        # it is not a classification-margin or probability estimate.
        match_contrib = 0.0
        if gesture_label in ("Open Palm", "Fist", "One Finger", "Two Fingers"):
            match_contrib = 25.0
        elif gesture_label == "Unknown":
            match_contrib = 8.0

        # 4. History consistency contribution (0 - 10 pts)
        history_contrib = min(10.0, max(0.0, history_confidence * 10.0))

        total_confidence = score_contrib + geom_contrib + match_contrib + history_contrib

        # Clamp strictly to 0.0 to 100.0
        return float(min(100.0, max(0.0, round(total_confidence, 1))))

    def process_gesture(
        self,
        contour: Optional[np.ndarray],
        contour_score: float = 0.0,
        history_confidence: float = 0.0,
    ) -> GestureResult:
        """Process contour and return detailed GestureResult."""
        if contour is None:
            return GestureResult(
                gesture=None,
                raw_gesture=None,
                confidence=0.0,
                is_hand_detected=False,
                features=None,
            )

        features = self.extract_features(contour)
        if features is None:
            return GestureResult(
                gesture=None,
                raw_gesture=None,
                confidence=0.0,
                is_hand_detected=False,
                features=None,
            )

        raw_gesture = self._classify_by_features(features)
        confidence = self.calculate_confidence(
            contour=contour,
            contour_score=contour_score,
            features=features,
            gesture_label=raw_gesture,
            history_confidence=history_confidence,
        )

        return GestureResult(
            gesture=raw_gesture,
            raw_gesture=raw_gesture,
            confidence=confidence,
            is_hand_detected=True,
            features=features,
        )

    def set_threshold(self, key: str, value: float) -> None:
        if key not in self.thresholds:
            raise KeyError(f"Unknown gesture threshold: {key}")
        self.thresholds[key] = value

    def update_thresholds(self, thresholds: Dict[str, float]) -> None:
        for key, value in thresholds.items():
            self.set_threshold(key, value)

    def get_threshold(self, key: str) -> Optional[float]:
        return self.thresholds.get(key)

    def get_all_thresholds(self) -> Dict[str, float]:
        return self.thresholds.copy()
