"""
Main application: Real-time hand gesture recognition with HSV-based skin detection.
Orchestrates camera, preprocessing, skin detection, hand detection, and gesture recognition.
"""

import os
import sys
from typing import Optional

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.calibration import CalibrationManager
from src.camera import CameraCapture
from src.gesture_history import GestureHistory
from src.gesture_recognition import GestureRecognizer
from src.hand_detection import HandDetector
from src.skin_detection import SkinDetector


class HandGestureApp:
    """Main application class orchestrating all components."""

    # Gesture to action mapping
    GESTURE_ACTIONS = {
        "Fist": "STOP",
        "Open Palm": "START",
        "One Finger": "SELECT",
        "Two Fingers": "NEXT",
    }

    def __init__(
        self,
        camera_id: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        resize_factor: float = 1.0,
        skip_calibration: bool = False,
    ) -> None:
        print("Hand Gesture Recognition System")
        print("=" * 50)

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.resize_factor = resize_factor

        try:
            self.camera = CameraCapture(
                camera_id=camera_id,
                target_width=frame_width,
                target_height=frame_height,
                resize_factor=resize_factor,
            )
            print("Camera initialized")
        except RuntimeError as e:
            print(f"Camera error: {e!s}")
            sys.exit(1)

        self.skin_detector = SkinDetector()
        print("Skin detector initialized")

        self.hand_detector = HandDetector(
            min_contour_area=500,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        print("Hand detector initialized")

        self.gesture_recognizer = GestureRecognizer()
        print("Gesture recognizer initialized")

        self.gesture_history = GestureHistory(buffer_size=8, consensus_threshold=4)
        print("Gesture history (temporal smoothing) initialized")

        self.skip_calibration = skip_calibration
        self.calibrated = False

        if not skip_calibration:
            self._handle_calibration()
        else:
            self.calibrated = True
            print("Calibration skipped (using defaults)")

        print("=" * 50)
        print("Ready to run. Press 'q' to quit, 'c' to recalibrate.")
        print("=" * 50)

    def _handle_calibration(self) -> None:
        """Check for saved calibration, or run interactive calibration."""
        calibration = CalibrationManager.load_calibration()
        if calibration:
            lower, upper = calibration
            skin_model = CalibrationManager.load_skin_model()
            self.skin_detector = SkinDetector(lower, upper, statistical_model=skin_model)
            self.calibrated = True
            print("Loaded saved calibration")
            return

        print("\nCALIBRATION MODE")
        print("-" * 50)
        self._run_calibration_interactive()

    def _run_calibration_interactive(self) -> None:
        """
        Interactive calibration: user places palm in center box and presses SPACE.
        """
        print("Instructions:")
        print("1. Place your PALM in the center box")
        print("2. Press SPACE to capture 10 valid samples")
        print("3. Keep hand steady")
        print("4. Press ESC to skip (use defaults)")
        print("-" * 50)

        samples = []
        sample_count = 0
        target_samples = 10

        while True:
            ret, frame = self.camera.get_frame()
            if not ret:
                print("Camera error during calibration")
                self.calibrated = False
                return

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Samples: {sample_count}/{target_samples}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            if sample_count > 0:
                cv2.putText(
                    frame,
                    "Sampling...",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Press SPACE to sample",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )

            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(30) & 0xFF

            if key == ord(" "):
                cy, cx = h // 2, w // 2
                region_size = 60
                y1, y2 = max(0, cy - region_size), min(h, cy + region_size)
                x1, x2 = max(0, cx - region_size), min(w, cx + region_size)

                # Inner crop to avoid edge contamination
                inset = 15
                sample_region = frame[y1 + inset : y2 - inset, x1 + inset : x2 - inset]
                roi_mask = np.full(sample_region.shape[:2], 255, dtype=np.uint8)

                valid_cnt = SkinDetector.valid_calibration_pixel_count(sample_region, roi_mask)
                total_cnt = sample_region.shape[0] * sample_region.shape[1]
                pct = (100.0 * valid_cnt / total_cnt) if total_cnt > 0 else 0.0

                if pct < 50.0:
                    print(f"Sample rejected ({pct:.1f}% valid). Place more palm inside the box.")
                    continue

                print(f"Sample {sample_count + 1} accepted: {valid_cnt}/{total_cnt} ({pct:.1f}%)")
                samples.append(sample_region)
                sample_count += 1

                if sample_count >= target_samples:
                    break

            elif key == 27:  # ESC
                print("Calibration skipped")
                self.calibrated = False
                cv2.destroyWindow("Calibration")
                return

        cv2.destroyWindow("Calibration")

        print("\nProcessing calibration samples...")
        lower, upper, stats = SkinDetector.calibrate_from_samples(samples)

        if not stats.get("calibration_valid", False):
            print("Calibration failed; please recalibrate.")
            self.calibrated = False
            return

        combined = np.vstack(samples)
        skin_model = SkinDetector.build_statistical_model(
            combined, np.ones(combined.shape[:2], dtype=np.uint8) * 255
        )

        self.skin_detector = SkinDetector(lower, upper, statistical_model=skin_model)
        self.calibrated = CalibrationManager.save_calibration(lower, upper, skin_model)

        print("Calibration complete!")
        print(f"  Lower HSV: {list(lower)}")
        print(f"  Upper HSV: {list(upper)}")

    def run(self) -> None:
        """Main event loop."""
        if not self.calibrated:
            print("Not calibrated. Exiting.")
            return

        try:
            while True:
                ret, frame = self.camera.get_frame()
                if not ret:
                    print("Camera dropped. Exiting.")
                    break

                skin_mask = self.skin_detector.detect_and_clean(frame, morphology_op="both")
                hand_contour = self.hand_detector.find_hand_contour(skin_mask)

                output_frame = frame.copy()

                if hand_contour is not None:
                    output_frame = self.hand_detector.draw_both(output_frame, hand_contour)
                    gesture = self.gesture_recognizer.recognize_gesture(hand_contour)
                    self.gesture_history.add_frame(gesture)

                    smoothed = self.gesture_history.get_smoothed_gesture()
                    action = self.GESTURE_ACTIONS.get(smoothed, "")

                    y_offset = 40
                    if smoothed == "Unknown":
                        cv2.putText(
                            output_frame,
                            "Unknown Gesture",
                            (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 165, 255),
                            2,
                        )
                    elif smoothed:
                        cv2.putText(
                            output_frame,
                            f"Gesture: {smoothed}",
                            (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            output_frame,
                            f"Action: {action}",
                            (20, y_offset + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            2,
                        )
                    else:
                        cv2.putText(
                            output_frame,
                            f"Raw: {gesture}",
                            (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (100, 100, 255),
                            2,
                        )
                else:
                    self.gesture_history.add_frame(None)
                    cv2.putText(
                        output_frame,
                        "No Hand Detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

                fps = self.camera.get_fps()
                cv2.putText(
                    output_frame,
                    f"FPS: {fps:.1f}",
                    (20, output_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )

                cv2.imshow("Hand Gesture Recognition", output_frame)

                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    cv2.destroyWindow("Hand Gesture Recognition")
                    self._run_calibration_interactive()

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up camera and openCV windows."""
        cv2.destroyAllWindows()
        self.camera.release()
        print("Cleanup complete")


def main() -> None:
    app = HandGestureApp()
    app.run()


if __name__ == "__main__":
    main()
