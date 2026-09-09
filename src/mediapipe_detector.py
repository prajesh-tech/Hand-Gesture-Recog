"""
MediaPipe 3D Hand Landmark Detector module.
Extracts 21 spatial hand landmark keypoints (x, y, z) and draws skeleton overlays.
Strictly decoupled: does NOT contain gesture classification logic.

Uses the MediaPipe Tasks HandLandmarker API (mediapipe >= 0.10.x, Python 3.13 compatible).
The legacy mediapipe.solutions.hands API was removed in MediaPipe 0.10.x and is unavailable
on Python 3.13; the Tasks API is the correct replacement.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from src.results import LandmarkDetectionResult

# Hand skeleton connections (21-landmark graph, same topology as legacy solutions API)
_HAND_CONNECTIONS: frozenset = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky + palm
])

# Resolve the bundled model relative to this file (models/ at repo root)
_DEFAULT_MODEL_PATH: str = os.path.join(
    os.path.dirname(__file__), "..", "models", "hand_landmarker.task"
)


class MediaPipeDetector:
    """
    Decoupled hand landmark detector using MediaPipe Tasks HandLandmarker API.
    Produces plain Python coordinate dictionaries for rule-based geometric evaluation.

    Replaces the legacy mediapipe.solutions.hands API which was removed in
    MediaPipe 0.10.x and is unavailable on Python 3.13.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 1,
        static_image_mode: bool = False,
        model_path: Optional[str] = None,
    ) -> None:
        """
        Initialize MediaPipe Tasks HandLandmarker pipeline.

        Args:
            min_detection_confidence: Minimum confidence for hand detection (default calibrated to 0.5).
            min_tracking_confidence: Minimum confidence for landmark tracking (default calibrated to 0.5).
            max_num_hands: Maximum number of hands to track (default 1).
            static_image_mode: When True uses IMAGE mode (single frames); False uses VIDEO mode
                               (temporal tracking across consecutive frames).
            model_path: Path to hand_landmarker.task model file. Defaults to models/hand_landmarker.task
                        at repo root.
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

        resolved_model = model_path or _DEFAULT_MODEL_PATH
        if not os.path.isfile(resolved_model):
            raise FileNotFoundError(
                f"HandLandmarker model not found at: {resolved_model}\n"
                "Download it with:\n"
                "  wget -O models/hand_landmarker.task \\\n"
                "    https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task"
            )

        running_mode = VisionTaskRunningMode.IMAGE if static_image_mode else VisionTaskRunningMode.VIDEO

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=resolved_model),
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hands = HandLandmarker.create_from_options(options)
        self._running_mode = running_mode
        self._frame_timestamp_ms: int = 0

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
            raise ValueError(
                f"Frame must be a non-empty 3-channel image, got shape "
                f"{frame.shape if hasattr(frame, 'shape') else 'invalid'}"
            )

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l_chan)
        merged_lab = cv2.merge((l_clahe, a_chan, b_chan))
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    def process_frame(
        self, frame: np.ndarray, draw_labels: bool = False
    ) -> Tuple[Optional[List[Dict[str, float]]], np.ndarray]:
        """
        Process an input BGR frame to extract 21 3D hand landmarks and render skeleton.

        Args:
            frame: Input BGR image (H, W, 3).
            draw_labels: When True, renders numerical keypoint indices (0-20) beside joints.

        Returns:
            Tuple of (landmarks, annotated_frame):
                - landmarks: List of 21 landmark dicts [{'id': int, 'x': float, 'y': float, 'z': float}, ...]
                             or None if no hand detected.
                - annotated_frame: Copy of frame with hand skeleton connections rendered.
        """
        self.validate_frame(frame)
        annotated_frame = frame.copy()

        # Convert BGR -> RGB and guarantee C-contiguous memory for the Tasks API
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not rgb_frame.flags.c_contiguous:
            rgb_frame = np.ascontiguousarray(rgb_frame)

        # Wrap in mp.Image (Tasks API requires this container)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Advance monotonic timestamp for VIDEO mode temporal tracking
        self._frame_timestamp_ms += 33  # ~30 fps cadence

        if self._running_mode == VisionTaskRunningMode.VIDEO:
            result = self.hands.detect_for_video(mp_image, self._frame_timestamp_ms)
        else:
            result = self.hands.detect(mp_image)

        if not result or not result.hand_landmarks:
            return None, annotated_frame

        # Take primary hand (first detected)
        primary_lms = result.hand_landmarks[0]  # List[NormalizedLandmark]

        # Draw skeleton overlay directly on annotated_frame (BGR)
        h, w = annotated_frame.shape[:2]
        for start_idx, end_idx in _HAND_CONNECTIONS:
            if start_idx < len(primary_lms) and end_idx < len(primary_lms):
                p1 = primary_lms[start_idx]
                p2 = primary_lms[end_idx]
                x1, y1 = int(p1.x * w), int(p1.y * h)
                x2, y2 = int(p2.x * w), int(p2.y * h)
                cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for idx, lm in enumerate(primary_lms):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated_frame, (cx, cy), 4, (0, 0, 255), -1)
            if draw_labels:
                cv2.putText(
                    annotated_frame,
                    str(idx),
                    (cx + 5, cy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 0),
                    1,
                )

        # Extract 21 landmarks into plain Python dictionaries
        landmarks: List[Dict[str, float]] = []
        for idx, lm in enumerate(primary_lms):
            landmarks.append({
                "id": int(idx),
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
            })

        # RC3 DIAGNOSTIC: Check wrist (Landmark 0) boundary proximity.
        # MediaPipe requires both the wrist joint and palm base to construct the
        # hand bounding box.  If the wrist is near the frame edge the bounding box
        # will be partially outside the image, causing detection loss on the next frame.
        EDGE_MARGIN: float = 0.10  # 10 % of normalised frame width/height
        wrist = landmarks[0]
        wrist_near_edge = (
            wrist["x"] < EDGE_MARGIN
            or wrist["x"] > (1.0 - EDGE_MARGIN)
            or wrist["y"] < EDGE_MARGIN
            or wrist["y"] > (1.0 - EDGE_MARGIN)
        )
        if wrist_near_edge:
            print(
                f"[MediaPipeDetector WARNING] Wrist (LM0) near frame edge "
                f"(x={wrist['x']:.2f}, y={wrist['y']:.2f}). "
                "Move hand toward the centre of the camera frame to avoid bounding-box loss."
            )

        return landmarks, annotated_frame

    def process_frame_to_result(
        self, frame: np.ndarray, draw_labels: bool = False
    ) -> LandmarkDetectionResult:
        """Convenience method returning structured LandmarkDetectionResult."""
        landmarks, annotated_frame = self.process_frame(frame, draw_labels=draw_labels)
        return LandmarkDetectionResult(
            landmarks=landmarks,
            annotated_frame=annotated_frame,
            is_hand_detected=landmarks is not None,
        )

    def close(self) -> None:
        """Release MediaPipe HandLandmarker resources."""
        if hasattr(self, "hands") and self.hands is not None:
            self.hands.close()

    def __enter__(self) -> "MediaPipeDetector":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
