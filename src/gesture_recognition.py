"""
Explainable 3D Hand Landmark Gesture Recognition with Heuristic Confidence Scoring.
Strategy 1: Evaluates 21 MediaPipe spatial keypoints using robust, orientation-aware
relative geometry (Euclidean distance ratios and vector dot product alignment).
STRICT SEPARATION OF CONCERNS: Operates purely on plain coordinate dictionaries.
Zero dependency on MediaPipe APIs or classes.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from src.results import GestureResult


class LandmarkGestureRecognizer:
    """
    Rule-based gesture recognizer operating purely on 21 3D hand landmarks.
    
    Landmark indexing convention (MediaPipe Hands):
      0: Wrist
      Thumb:  1: CMC, 2: MCP,  3: IP,   4: TIP
      Index:  5: MCP, 6: PIP,  7: DIP,  8: TIP
      Middle: 9: MCP, 10: PIP, 11: DIP, 12: TIP
      Ring:   13: MCP, 14: PIP, 15: DIP, 16: TIP
      Pinky:  17: MCP, 18: PIP, 19: DIP, 20: TIP
    """

    # Anatomical index references
    WRIST = 0
    THUMB_TIP = 4
    INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
    RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
    PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

    DEFAULT_THRESHOLDS = {
        # Finger extension Euclidean ratio threshold: dist(Wrist, Tip) / dist(Wrist, PIP)
        "extension_ratio_index": 1.35,
        "extension_ratio_middle": 1.35,
        "extension_ratio_ring": 1.35,
        "extension_ratio_pinky": 1.35,
        # Finger alignment/straightness threshold: cos(theta) between (MCP->PIP) and (PIP->Tip)
        "straightness_threshold": 0.85,
        # Thumb spread ratio threshold: dist(Thumb_Tip, Pinky_MCP) / dist(Wrist, Pinky_MCP)
        "thumb_spread_threshold": 0.85,
        # Landmark structural plausibility thresholds
        "min_coord_variance": 0.001,
        "min_palm_size": 0.04,
    }

    def __init__(self, thresholds: Optional[Dict[str, float]] = None) -> None:
        """
        Initialize recognizer with configurable geometric thresholds.
        """
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)

    @staticmethod
    def _dist_3d(p1: Dict[str, float], p2: Dict[str, float]) -> float:
        """Calculate 3D Euclidean distance between two landmark points."""
        dx = p1.get("x", 0.0) - p2.get("x", 0.0)
        dy = p1.get("y", 0.0) - p2.get("y", 0.0)
        dz = p1.get("z", 0.0) - p2.get("z", 0.0)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _vector_cos_angle(
        v1: Tuple[float, float, float], v2: Tuple[float, float, float]
    ) -> float:
        """Compute cosine of angle between two 3D vectors."""
        dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        norm1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])
        norm2 = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
        if norm1 < 1e-7 or norm2 < 1e-7:
            return 0.0
        val = dot / (norm1 * norm2)
        return max(-1.0, min(1.0, val))

    def _get_landmark_map(
        self, landmarks: List[Dict[str, float]]
    ) -> Optional[Dict[int, Dict[str, float]]]:
        """Convert list of landmarks to an ID-keyed dictionary for O(1) indexed lookup."""
        if not isinstance(landmarks, (list, tuple)) or len(landmarks) < 21:
            return None
        lm_map = {}
        for idx, lm in enumerate(landmarks):
            if not isinstance(lm, dict):
                return None
            lm_id = lm.get("id", idx)
            lm_map[lm_id] = lm
        if len(lm_map) < 21:
            return None
        return lm_map

    def evaluate_finger(
        self,
        lm_map: Dict[int, Dict[str, float]],
        mcp_id: int,
        pip_id: int,
        tip_id: int,
        finger_name: str,
    ) -> Dict[str, Any]:
        """
        Evaluate extension state and geometric straightness of a single finger.
        
        A finger is extended if BOTH conditions are met:
        1. Euclidean Extension Ratio: dist(Wrist, Tip) / dist(Wrist, PIP) > THRESHOLD
        2. Alignment/Straightness: cos(theta) of (MCP->PIP) and (PIP->Tip) > THRESHOLD
        """
        wrist = lm_map[self.WRIST]
        mcp = lm_map[mcp_id]
        pip = lm_map[pip_id]
        tip = lm_map[tip_id]

        dist_wrist_tip = self._dist_3d(wrist, tip)
        dist_wrist_pip = self._dist_3d(wrist, pip)

        ratio = (
            dist_wrist_tip / dist_wrist_pip
            if dist_wrist_pip > 1e-7
            else 0.0
        )

        v1 = (pip["x"] - mcp["x"], pip["y"] - mcp["y"], pip.get("z", 0.0) - mcp.get("z", 0.0))
        v2 = (tip["x"] - pip["x"], tip["y"] - pip["y"], tip.get("z", 0.0) - pip.get("z", 0.0))
        straightness = self._vector_cos_angle(v1, v2)

        ratio_thresh = self.thresholds.get(f"extension_ratio_{finger_name}", 1.35)
        straightness_thresh = self.thresholds.get("straightness_threshold", 0.85)

        is_extended = (ratio > ratio_thresh) and (straightness > straightness_thresh)

        return {
            "is_extended": bool(is_extended),
            "ratio": float(ratio),
            "ratio_threshold": float(ratio_thresh),
            "ratio_margin": float(ratio - ratio_thresh),
            "straightness": float(straightness),
            "straightness_threshold": float(straightness_thresh),
            "straightness_margin": float(straightness - straightness_thresh),
        }

    def evaluate_thumb(
        self, lm_map: Dict[int, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Evaluate normalized thumb extension spread:
        thumb_ratio = dist(Thumb_Tip, Pinky_MCP) / dist(Wrist, Pinky_MCP)
        Thumb is extended if thumb_ratio > thumb_spread_threshold.
        """
        wrist = lm_map[self.WRIST]
        thumb_tip = lm_map[self.THUMB_TIP]
        pinky_mcp = lm_map[self.PINKY_MCP]

        dist_thumb_pinky = self._dist_3d(thumb_tip, pinky_mcp)
        dist_wrist_pinky = self._dist_3d(wrist, pinky_mcp)

        thumb_ratio = (
            dist_thumb_pinky / dist_wrist_pinky
            if dist_wrist_pinky > 1e-7
            else 0.0
        )

        thresh = self.thresholds.get("thumb_spread_threshold", 0.85)
        is_extended = thumb_ratio > thresh

        return {
            "is_extended": bool(is_extended),
            "ratio": float(thumb_ratio),
            "ratio_threshold": float(thresh),
            "ratio_margin": float(thumb_ratio - thresh),
        }

    def extract_features(
        self, landmarks: Optional[List[Dict[str, float]]]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract spatial geometric features and finger extension states from 21 landmarks.
        Returns None if landmarks are missing, invalid, or fewer than 21 keypoints.
        """
        if landmarks is None:
            return None

        lm_map = self._get_landmark_map(landmarks)
        if lm_map is None:
            return None

        # Finger evaluations
        index_eval = self.evaluate_finger(
            lm_map, self.INDEX_MCP, self.INDEX_PIP, self.INDEX_TIP, "index"
        )
        middle_eval = self.evaluate_finger(
            lm_map, self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_TIP, "middle"
        )
        ring_eval = self.evaluate_finger(
            lm_map, self.RING_MCP, self.RING_PIP, self.RING_TIP, "ring"
        )
        pinky_eval = self.evaluate_finger(
            lm_map, self.PINKY_MCP, self.PINKY_PIP, self.PINKY_TIP, "pinky"
        )
        thumb_eval = self.evaluate_thumb(lm_map)

        # Coordinate statistics for landmark quality assessment
        xs = [lm["x"] for lm in lm_map.values()]
        ys = [lm["y"] for lm in lm_map.values()]
        zs = [lm.get("z", 0.0) for lm in lm_map.values()]

        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        var_x = sum((x - mean_x) ** 2 for x in xs) / len(xs)
        var_y = sum((y - mean_y) ** 2 for y in ys) / len(ys)
        coord_var = (var_x + var_y) / 2.0

        palm_size = self._dist_3d(lm_map[self.WRIST], lm_map[self.MIDDLE_MCP])

        extended_finger_count = sum(
            1 for f in [index_eval, middle_eval, ring_eval, pinky_eval] if f["is_extended"]
        )

        return {
            "thumb": thumb_eval,
            "index": index_eval,
            "middle": middle_eval,
            "ring": ring_eval,
            "pinky": pinky_eval,
            "extended_finger_count": extended_finger_count,
            "thumb_extended": thumb_eval["is_extended"],
            "coord_variance": float(coord_var),
            "palm_size": float(palm_size),
        }

    def classify_gesture(self, features: Dict[str, Any]) -> str:
        """
        Classify spatial hand layout into one of the four supported gestures, or Unknown.
        
        Rules:
        - Fist: All 4 fingers closed & thumb tucked.
        - Open Palm: All 4 fingers extended & thumb extended.
        - One Finger: Only Index extended; Middle, Ring, Pinky closed.
        - Two Fingers: Index and Middle extended; Ring, Pinky closed.
        - Unknown: Anything else.
        """
        thumb = features["thumb"]["is_extended"]
        index = features["index"]["is_extended"]
        middle = features["middle"]["is_extended"]
        ring = features["ring"]["is_extended"]
        pinky = features["pinky"]["is_extended"]

        # 1. Fist: All 4 fingers closed & thumb tucked
        if not thumb and not index and not middle and not ring and not pinky:
            return "Fist"

        # 2. Open Palm: All 4 fingers extended & thumb extended
        if thumb and index and middle and ring and pinky:
            return "Open Palm"

        # 3. One Finger: Index extended; Middle, Ring, Pinky closed
        if index and not middle and not ring and not pinky:
            return "One Finger"

        # 4. Two Fingers: Index & Middle extended; Ring, Pinky closed
        if index and middle and not ring and not pinky:
            return "Two Fingers"

        return "Unknown"

    def recognize_gesture(
        self, landmarks: Optional[List[Dict[str, float]]]
    ) -> Optional[str]:
        """
        Recognize gesture from landmarks list.
        Returns None when no hand is present.
        """
        features = self.extract_features(landmarks)
        if features is None:
            return None
        return self.classify_gesture(features)

    def calculate_confidence(
        self,
        landmarks: Optional[List[Dict[str, float]]],
        features: Optional[Dict[str, Any]],
        gesture_label: Optional[str],
        history_confidence: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate deterministic 0–100% heuristic confidence score:
        Total = clamp(Quality (0–30) + Rule Match (0–50) + Stability (0–20), 0, 100)

        Returns:
            Tuple of (total_confidence, breakdown_dict)
        """
        if landmarks is None or features is None or gesture_label is None:
            return 0.0, {"quality": 0.0, "rule_match": 0.0, "stability": 0.0}

        # 1. Landmark Quality (0–30 pts)
        # Structural plausibility: coordinate variance and anatomical proportions
        quality_score = 0.0
        var = features.get("coord_variance", 0.0)
        min_var = self.thresholds.get("min_coord_variance", 0.001)
        palm_sz = features.get("palm_size", 0.0)
        min_palm = self.thresholds.get("min_palm_size", 0.04)

        if var >= min_var:
            # Scaled contribution up to 15 pts
            quality_score += min(15.0, 10.0 + (var / (min_var * 10.0)) * 5.0)
        else:
            quality_score += max(0.0, (var / max(min_var, 1e-7)) * 10.0)

        if palm_sz >= min_palm:
            quality_score += min(15.0, 10.0 + (palm_sz / (min_palm * 2.0)) * 5.0)
        else:
            quality_score += max(0.0, (palm_sz / max(min_palm, 1e-7)) * 10.0)

        quality_score = min(30.0, max(0.0, quality_score))

        # 2. Rule Match Margin (0–50 pts)
        # Measures how decisively spatial ratios exceed or fall below classification thresholds
        rule_match_score = 0.0
        if gesture_label in ("Fist", "Open Palm", "One Finger", "Two Fingers"):
            base_rule_pts = 30.0
            margin_pts = 0.0

            # Target states for each digit per gesture
            # (thumb, index, middle, ring, pinky)
            target_states = {
                "Fist": (False, False, False, False, False),
                "Open Palm": (True, True, True, True, True),
                "One Finger": (None, True, False, False, False),
                "Two Fingers": (None, True, True, False, False),
            }
            targets = target_states[gesture_label]

            # Evaluate 4 fingers
            finger_keys = ["index", "middle", "ring", "pinky"]
            for i, f_key in enumerate(finger_keys):
                t_state = targets[i + 1]
                f_eval = features[f_key]
                if t_state is True:
                    # Extended: reward positive ratio margin and straightness margin
                    r_margin = max(0.0, f_eval["ratio_margin"])
                    s_margin = max(0.0, f_eval["straightness_margin"])
                    margin_pts += min(3.5, r_margin * 5.0) + min(1.5, s_margin * 5.0)
                elif t_state is False:
                    # Closed: reward negative ratio margin (distance below threshold)
                    closed_margin = max(0.0, -f_eval["ratio_margin"])
                    margin_pts += min(5.0, closed_margin * 5.0)

            # Evaluate thumb if constrained
            if targets[0] is not None:
                t_eval = features["thumb"]
                if targets[0] is True:
                    t_margin = max(0.0, t_eval["ratio_margin"])
                    margin_pts += min(5.0, t_margin * 10.0)
                else:
                    t_closed_margin = max(0.0, -t_eval["ratio_margin"])
                    margin_pts += min(5.0, t_closed_margin * 10.0)
            else:
                margin_pts += 3.0  # unconstrained thumb neutral points

            rule_match_score = min(50.0, base_rule_pts + min(20.0, margin_pts))

        elif gesture_label == "Unknown":
            rule_match_score = 10.0

        # 3. Temporal Stability (0–20 pts)
        # Agreement within the history buffer
        stability_score = min(20.0, max(0.0, history_confidence * 20.0))

        total_confidence = quality_score + rule_match_score + stability_score
        clamped_confidence = float(min(100.0, max(0.0, round(total_confidence, 1))))

        breakdown = {
            "quality": round(quality_score, 1),
            "rule_match": round(rule_match_score, 1),
            "stability": round(stability_score, 1),
        }

        return clamped_confidence, breakdown

    def process_gesture(
        self,
        landmarks: Optional[List[Dict[str, float]]],
        history_confidence: float = 0.0,
    ) -> GestureResult:
        """
        Process landmarks and produce structured GestureResult.
        """
        if landmarks is None:
            return GestureResult(
                gesture=None,
                raw_gesture=None,
                confidence=0.0,
                is_hand_detected=False,
                features=None,
                confidence_breakdown={"quality": 0.0, "rule_match": 0.0, "stability": 0.0},
            )

        features = self.extract_features(landmarks)
        if features is None:
            return GestureResult(
                gesture=None,
                raw_gesture=None,
                confidence=0.0,
                is_hand_detected=False,
                features=None,
                confidence_breakdown={"quality": 0.0, "rule_match": 0.0, "stability": 0.0},
            )

        raw_gesture = self.classify_gesture(features)
        confidence, breakdown = self.calculate_confidence(
            landmarks=landmarks,
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
            confidence_breakdown=breakdown,
        )

    def set_threshold(self, key: str, value: float) -> None:
        """Set a single geometric threshold."""
        if key not in self.thresholds:
            raise KeyError(f"Unknown gesture threshold: {key}")
        self.thresholds[key] = float(value)

    def update_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Update multiple geometric thresholds."""
        for key, value in thresholds.items():
            self.set_threshold(key, value)

    def get_threshold(self, key: str) -> Optional[float]:
        """Get the value of a geometric threshold."""
        return self.thresholds.get(key)

    def get_all_thresholds(self) -> Dict[str, float]:
        """Return a copy of all current thresholds."""
        return self.thresholds.copy()


# Alias for backward compatibility with existing imports
GestureRecognizer = LandmarkGestureRecognizer
