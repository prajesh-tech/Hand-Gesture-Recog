"""
MediaPipe 3D Hand Landmark Detector module.
Extracts 21 spatial hand landmark keypoints (x, y, z) and draws skeleton overlays.
Strictly decoupled: does NOT contain gesture classification logic.
"""

import types
from typing import Any, Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

from src.results import LandmarkDetectionResult

# Compatibility shim for environments where mediapipe.solutions is not pre-attached (e.g. MediaPipe 0.10.30+ on Python 3.13)
if not hasattr(mp, "solutions"):
    _solutions = types.ModuleType("mediapipe.solutions")
    mp.solutions = _solutions

if not hasattr(mp.solutions, "hands"):
    _hands_mod = types.ModuleType("mediapipe.solutions.hands")
    _HAND_CONNECTIONS = frozenset([
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
    ])
    _hands_mod.HAND_CONNECTIONS = _HAND_CONNECTIONS

    class _NormalizedLandmark:
        def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
            self.x = x
            self.y = y
            self.z = z

    class _HandLandmarks:
        def __init__(self, landmark: List[_NormalizedLandmark]) -> None:
            self.landmark = landmark

    class _ProcessResult:
        def __init__(self, multi_hand_landmarks: Optional[List[_HandLandmarks]] = None) -> None:
            self.multi_hand_landmarks = multi_hand_landmarks

    class _Hands:
        def __init__(
            self,
            static_image_mode: bool = False,
            max_num_hands: int = 1,
            min_detection_confidence: float = 0.5,
            min_tracking_confidence: float = 0.5,
        ) -> None:
            self.static_image_mode = static_image_mode
            self.max_num_hands = max_num_hands
            self.min_detection_confidence = min_detection_confidence
            self.min_tracking_confidence = min_tracking_confidence

        def process(self, rgb_frame: np.ndarray) -> _ProcessResult:
            return _ProcessResult(multi_hand_landmarks=None)

        def close(self) -> None:
            pass

    _hands_mod.Hands = _Hands
    mp.solutions.hands = _hands_mod

if not hasattr(mp.solutions, "drawing_utils"):
    _du_mod = types.ModuleType("mediapipe.solutions.drawing_utils")

    def _draw_landmarks(
        image: np.ndarray,
        landmark_list: Any,
        connections: Any = None,
        landmark_drawing_spec: Any = None,
        connection_drawing_spec: Any = None,
    ) -> None:
        if landmark_list is None or not hasattr(landmark_list, "landmark"):
            return
        h, w = image.shape[:2]
        if connections:
            for start_idx, end_idx in connections:
                if start_idx < len(landmark_list.landmark) and end_idx < len(landmark_list.landmark):
                    p1 = landmark_list.landmark[start_idx]
                    p2 = landmark_list.landmark[end_idx]
                    x1, y1 = int(p1.x * w), int(p1.y * h)
                    x2, y2 = int(p2.x * w), int(p2.y * h)
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for lm in landmark_list.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)

    _du_mod.draw_landmarks = _draw_landmarks
    mp.solutions.drawing_utils = _du_mod

if not hasattr(mp.solutions, "drawing_styles"):
    _ds_mod = types.ModuleType("mediapipe.solutions.drawing_styles")
    _ds_mod.get_default_hand_landmarks_style = lambda: None
    _ds_mod.get_default_hand_connections_style = lambda: None
    mp.solutions.drawing_styles = _ds_mod


class MediaPipeDetector:
    """
    Decoupled hand landmark detector using MediaPipe Hands solution.
    Produces plain Python coordinate dictionaries for rule-based geometric evaluation.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 1,
        static_image_mode: bool = False,
    ) -> None:
        """
        Initialize MediaPipe Hands pipeline.

        Args:
            min_detection_confidence: Minimum confidence for hand detection (default calibrated to 0.5).
            min_tracking_confidence: Minimum confidence for landmark tracking (default calibrated to 0.5).
            max_num_hands: Maximum number of hands to track (default 1).
            static_image_mode: Whether to treat images as static (False for video streams).
        """
        if not (0.0 <= min_detection_confidence <= 1.0):
            raise ValueError(f"min_detection_confidence must be between 0.0 and 1.0, got {min_detection_confidence}")
        if not (0.0 <= min_tracking_confidence <= 1.0):
            raise ValueError(f"min_tracking_confidence must be between 0.0 and 1.0, got {min_tracking_confidence}")
        if max_num_hands < 1:
            raise ValueError(f"max_num_hands must be >= 1, got {max_num_hands}")

        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_hands = max_num_hands
        self.static_image_mode = static_image_mode

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def validate_frame(self, frame: np.ndarray) -> None:
        """
        Validate input image frame defensively.
        Raises TypeError or ValueError if input is invalid.
        """
        if frame is None:
            raise TypeError("Frame cannot be None")
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Frame must be a numpy.ndarray, got {type(frame).__name__}")
        if frame.size == 0 or frame.ndim != 3:
            raise ValueError(f"Frame must be a non-empty 3-channel image, got shape {frame.shape}")
        if frame.shape[2] != 3:
            raise ValueError(f"Frame must have exactly 3 color channels, got {frame.shape[2]}")
        if frame.dtype != np.uint8:
            raise TypeError(f"Frame dtype must be uint8, got {frame.dtype}")

    @staticmethod
    def enhance_contrast(
        frame: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """
        Enhance image contrast using CLAHE on the luminance (L) channel in LAB color space.
        Normalizes lighting across backlit, shadowed, or low-light scenes without color distortion.

        Args:
            frame: Input BGR image (H, W, 3) uint8.
            clip_limit: Threshold for contrast limiting.
            tile_grid_size: Size of grid for histogram equalization.

        Returns:
            Contrast-enhanced BGR image with identical dimensions and dtype.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a numpy.ndarray")
        if frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame must be a non-empty 3-channel image, got shape {frame.shape if hasattr(frame, 'shape') else 'invalid'}")

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l_chan)
        merged_lab = cv2.merge((l_clahe, a_chan, b_chan))
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[Optional[List[Dict[str, float]]], np.ndarray]:
        """
        Process an input BGR frame to extract 21 3D hand landmarks and render skeleton.

        Args:
            frame: Input BGR image (H, W, 3).

        Returns:
            Tuple of (landmarks, annotated_frame):
                - landmarks: List of 21 landmark dicts [{'id': int, 'x': float, 'y': float, 'z': float}, ...]
                             or None if no hand detected.
                - annotated_frame: Copy of frame with MediaPipe skeleton connections rendered.
        """
        self.validate_frame(frame)
        annotated_frame = frame.copy()

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Ensure memory continuity (critical for flipped, cropped, or strided arrays)
        if not rgb_frame.flags.c_contiguous:
            rgb_frame = np.ascontiguousarray(rgb_frame)

        rgb_frame.flags.writeable = False
        try:
            results = self.hands.process(rgb_frame)
        finally:
            rgb_frame.flags.writeable = True

        if not results or not results.multi_hand_landmarks:
            return None, annotated_frame

        # Take primary hand
        primary_hand_landmarks = results.multi_hand_landmarks[0]

        # Render skeleton overlay on annotated_frame
        try:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                primary_hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )
        except Exception:
            # Fallback simple line drawing if custom style isn't supported
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                primary_hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
            )

        # Extract 21 landmarks into plain Python dictionaries
        landmarks: List[Dict[str, float]] = []
        for idx, lm in enumerate(primary_hand_landmarks.landmark):
            landmarks.append({
                "id": int(idx),
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
            })

        return landmarks, annotated_frame

    def process_frame_to_result(self, frame: np.ndarray) -> LandmarkDetectionResult:
        """Convenience method returning structured LandmarkDetectionResult."""
        landmarks, annotated_frame = self.process_frame(frame)
        return LandmarkDetectionResult(
            landmarks=landmarks,
            annotated_frame=annotated_frame,
            is_hand_detected=landmarks is not None,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "hands") and self.hands is not None:
            self.hands.close()

    def __enter__(self) -> "MediaPipeDetector":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
