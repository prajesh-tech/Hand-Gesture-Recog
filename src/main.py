"""
Main application: Real-time hand gesture recognition with MediaPipe 3D Landmark Tracking.
Strategy 1: Modular pipeline orchestrating camera, MediaPipe keypoint tracking,
rule-based geometric classification, temporal smoothing, and explainable confidence scoring.
"""

import os
import sys
from typing import Optional

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.camera import CameraCapture
from src.gesture_history import GestureHistory
from src.gesture_recognition import LandmarkGestureRecognizer
from src.mediapipe_detector import MediaPipeDetector
from src.results import DetectionFrameResult, LandmarkDetectionResult


class HandGestureApp:
    """Main application class orchestrating the MediaPipe 3D landmark gesture pipeline."""

    def __init__(
        self,
        camera_id: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        resize_factor: float = 1.0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        history_buffer_size: int = 8,
        consensus_threshold: int = 4,
        mirror: bool = True,
        auto_contrast: bool = False,
        debug: bool = True,
    ) -> None:
        print("Hand Gesture Recognition System (MediaPipe 3D Landmark Strategy 1)")
        print("=" * 65)

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.resize_factor = resize_factor
        self.debug = debug
        self.frame_count = 0

        try:
            self.camera = CameraCapture(
                camera_id=camera_id,
                target_width=frame_width,
                target_height=frame_height,
                resize_factor=resize_factor,
                mirror=mirror,
                auto_contrast=auto_contrast,
            )
            print("✓ Camera initialized successfully")
        except (RuntimeError, ValueError, TypeError) as e:
            print(f"✗ Camera error: {e!s}")
            sys.exit(1)

        self.detector = MediaPipeDetector(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=1,
            static_image_mode=False,
        )
        print("✓ MediaPipe 3D Landmark Detector initialized (conf=0.5)")

        self.gesture_recognizer = LandmarkGestureRecognizer()
        print("✓ Rule-based geometric Gesture Recognizer initialized")

        self.gesture_history = GestureHistory(
            buffer_size=history_buffer_size,
            consensus_threshold=consensus_threshold,
        )
        print("✓ Temporal consensus engine initialized")

        print("=" * 65)
        print("Ready to run. Press 'q' in video window or Ctrl+C to quit.")
        print("=" * 65)

    def process_current_frame(self) -> Optional[DetectionFrameResult]:
        """
        Execute explicit sequential processing pipeline:
        Raw Frame -> Landmarks -> Raw Gesture -> History -> Stable Gesture -> Confidence -> Display
        """
        ret, frame = self.camera.get_frame()
        if not ret or frame is None:
            return None

        self.frame_count += 1
        fps = self.camera.get_fps()

        # Step 1: MediaPipe 3D Landmark Detection & Skeleton Overlay
        landmarks, annotated_frame = self.detector.process_frame(frame)
        is_hand_detected = landmarks is not None

        # Step 2: Extract spatial geometric features and raw gesture classification
        gesture_res = self.gesture_recognizer.process_gesture(
            landmarks=landmarks,
            history_confidence=0.0,
        )

        # Step 3: Temporal smoothing and consensus voting
        raw_gesture = gesture_res.raw_gesture if is_hand_detected else None
        history_res = self.gesture_history.process_history(raw_gesture)

        # Step 4: Recalculate 0–100% confidence including current frame's contribution to history
        updated_history_confidence = self.gesture_history.get_confidence(raw_gesture)
        confidence, breakdown = self.gesture_recognizer.calculate_confidence(
            landmarks=landmarks,
            features=gesture_res.features,
            gesture_label=gesture_res.raw_gesture,
            history_confidence=updated_history_confidence,
        )
        gesture_res.confidence = confidence
        gesture_res.confidence_breakdown = breakdown

        landmark_res = LandmarkDetectionResult(
            landmarks=landmarks,
            annotated_frame=annotated_frame,
            is_hand_detected=is_hand_detected,
        )

        # Lightweight per-frame debug print
        if self.debug:
            lm_count = len(landmarks) if landmarks is not None else 0
            print(
                f"[DEBUG Frame {self.frame_count:04d}] HandDetected={is_hand_detected} "
                f"| Landmarks={lm_count:02d} | Raw={raw_gesture or 'None'} "
                f"| Conf={confidence:.0f}% | FPS={fps:.1f}"
            )

        return DetectionFrameResult(
            frame=frame,
            fps=fps,
            gesture_res=gesture_res,
            history_res=history_res,
            landmark_res=landmark_res,
            annotated_frame=annotated_frame,
        )

    def render_overlay(self, diag: DetectionFrameResult) -> np.ndarray:
        """Render diagnostic and action HUD on the annotated frame."""
        output_frame = diag.annotated_frame.copy() if diag.annotated_frame is not None else diag.frame.copy()

        smoothed = diag.history_res.smoothed_gesture
        action = diag.history_res.action
        confidence = diag.gesture_res.confidence
        raw_gesture = diag.gesture_res.raw_gesture
        is_detected = diag.gesture_res.is_hand_detected

        # Diagnostic HUD Line 1: Hand Detected True/False
        detect_color = (0, 255, 0) if is_detected else (0, 0, 255)
        cv2.putText(
            output_frame,
            f"Hand Detected: {is_detected}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            detect_color,
            2,
        )

        # Diagnostic HUD Line 2: Tracking Confidence
        cv2.putText(
            output_frame,
            f"Tracking Confidence: {confidence:.0f}%",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        y_offset = 100
        if is_detected:
            if smoothed:
                # Stable recognized gesture
                cv2.putText(
                    output_frame,
                    f"Gesture: {smoothed}",
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    output_frame,
                    f"Action: {action}",
                    (20, y_offset + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 255),
                    2,
                )
            elif raw_gesture == "Unknown":
                cv2.putText(
                    output_frame,
                    f"Gesture: Unknown",
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2,
                )
            else:
                # Raw gesture pending temporal consensus
                cv2.putText(
                    output_frame,
                    f"Raw: {raw_gesture} [Stabilizing]",
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (100, 100, 255),
                    2,
                )
        else:
            cv2.putText(
                output_frame,
                "Status: No hand in view (Position hand in camera frame)",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (180, 180, 180),
                2,
            )

        # Render FPS at bottom-left
        cv2.putText(
            output_frame,
            f"FPS: {diag.fps:.1f}",
            (20, output_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        return output_frame

    def run(self) -> None:
        """Main event loop."""
        try:
            while True:
                diag = self.process_current_frame()
                if diag is None:
                    print("Camera feed dropped. Exiting.")
                    break

                output_frame = self.render_overlay(diag)
                cv2.imshow("Hand Gesture Recognition", output_frame)

                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    break

        except KeyboardInterrupt:
            print("\nApplication interrupted by user.")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Release camera, detector, and window resources safely."""
        cv2.destroyAllWindows()
        if hasattr(self, "detector"):
            self.detector.close()
        if hasattr(self, "camera"):
            self.camera.release()
        print("✓ Cleanup complete. Session terminated.")


def main() -> None:
    app = HandGestureApp()
    app.run()


if __name__ == "__main__":
    main()
