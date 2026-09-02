"""
Type-safe result structures for HGR V1 pipeline stages.
Provides structured data passing between camera, skin detection, hand detection,
gesture recognition, history, and application UI.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CalibrationResult:
    """Result of calibration attempt or load operation."""
    success: bool
    status: str  # "SUCCESS", "FAILED", "CANCELLED", "RETAINED"
    lower_hsv: Optional[np.ndarray] = None
    upper_hsv: Optional[np.ndarray] = None
    message: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkinDetectionResult:
    """Result of HSV skin detection and morphological cleaning."""
    mask_raw: np.ndarray
    mask_clean: np.ndarray
    raw_pixel_count: int
    clean_pixel_count: int
    raw_percentage: float
    clean_percentage: float


@dataclass
class HandDetectionResult:
    """Result of hand contour detection and filtering."""
    selected_contour: Optional[np.ndarray]
    score: float = 0.0
    area: float = 0.0
    perimeter: float = 0.0
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    is_hand_detected: bool = False


@dataclass
class GestureResult:
    """Result of geometric gesture classification and confidence evaluation."""
    gesture: Optional[str]  # e.g., "Open Palm", "Fist", "One Finger", "Two Fingers", "Unknown", None
    raw_gesture: Optional[str]
    confidence: float  # Heuristic confidence score 0.0 to 100.0
    is_hand_detected: bool
    features: Optional[Dict[str, Any]] = None


@dataclass
class GestureHistoryResult:
    """Result of temporal smoothing over gesture history."""
    smoothed_gesture: Optional[str]
    action: str
    history: List[Optional[str]]
    temporal_confidence: float  # Ratio of history matching smoothed gesture (0.0 to 1.0)


@dataclass
class DetectionFrameResult:
    """Complete diagnostic result for a single processed video frame."""
    frame: np.ndarray
    fps: float
    skin_res: SkinDetectionResult
    hand_res: HandDetectionResult
    gesture_res: GestureResult
    history_res: GestureHistoryResult
