"""
Main application: Real-time hand gesture recognition with HSV-based skin detection.
Orchestrates camera, preprocessing, skin detection, hand detection, gesture recognition, and confidence scoring.
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
from src.results import CalibrationResult, DetectionFrameResult
from src.skin_detection import SkinDetector


class HandGestureApp:
    """Main application class orchestrating all components."""

    
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
        except (RuntimeError, ValueError, TypeError) as e:
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
            self.skin_detector = SkinDetector(lower, upper)
            self.calibrated = True
            print("Loaded saved calibration")
            return

        print("\nCALIBRATION MODE")
        print("-" * 50)
        cal_res = self._run_calibration_interactive()

        if cal_res.success and cal_res.lower_hsv is not None and cal_res.upper_hsv is not None:
            self.skin_detector = SkinDetector(cal_res.lower_hsv, cal_res.upper_hsv)
            saved = CalibrationManager.save_calibration(cal_res.lower_hsv, cal_res.upper_hsv)
            self.calibrated = saved
            print(f"✓ Calibration complete! Lower: {list(cal_res.lower_hsv)}, Upper: {list(cal_res.upper_hsv)}")
        else:
            self.calibrated = False
            print(f"✗ Initial calibration failed/cancelled ({cal_res.status}). Cannot start gesture detection.")

    def _run_calibration_interactive(self) -> CalibrationResult:
        """
        Interactive calibration: user places palm in center box and presses SPACE.
        """
        print("Instructions:")
        print("1. Place your PALM in the center box")
        print("2. Press SPACE to capture 10 valid samples")
        print("3. Keep hand steady")
        print("4. Press ESC to cancel")
        print("-" * 50)

        samples = []
        sample_count = 0
        target_samples = 10

        while True:
            ret, frame = self.camera.get_frame()
            if not ret or frame is None:
                print("Camera error during calibration")
                try:
                    cv2.destroyWindow("Calibration")
                except Exception:
                    pass
                return CalibrationResult(
                    success=False,
                    status="FAILED",
                    message="Camera read error during calibration",
                )

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
                    "Press SPACE to sample | ESC to cancel",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )

            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(30) & 0xFF

            if key == ord(" "):
                # Sample region matches the displayed green rectangle exactly:
                # same corners as the cv2.rectangle drawn at (w//4, h//4)→(3*w//4, 3*h//4).
                x1, y1 = w // 4, h // 4
                x2, y2 = 3 * w // 4, 3 * h // 4

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
                print("Calibration cancelled by user")
                try:
                    cv2.destroyWindow("Calibration")
                except Exception:
                    pass
                return CalibrationResult(
                    success=False,
                    status="CANCELLED",
                    message="Calibration cancelled by user",
                )

        try:
            cv2.destroyWindow("Calibration")
        except Exception:
            pass

        print("\nProcessing calibration samples...")
        lower, upper, stats = SkinDetector.calibrate_from_samples(samples)

        if not stats.get("calibration_valid", False):
            error_msg = stats.get("error", "Invalid bounds computed from samples")
            print(f"Calibration failed: {error_msg}")
            return CalibrationResult(
                success=False,
                status="FAILED",
                message=error_msg,
                diagnostics=stats,
            )

        return CalibrationResult(
            success=True,
            status="SUCCESS",
            lower_hsv=lower,
            upper_hsv=upper,
            message="Calibration successful",
            diagnostics=stats,
        )

    def process_current_frame(self) -> Optional[DetectionFrameResult]:
        """Capture and process a single frame through the full pipeline."""
        ret, frame = self.camera.get_frame()
        if not ret or frame is None:
            return None

        fps = self.camera.get_fps()

        # Step 1: Skin Detection
        skin_res = self.skin_detector.process_frame(frame, morphology_op="both")

        # Step 2: Hand Detection
        hand_res = self.hand_detector.process_mask(skin_res.mask_clean)

        # Step 3: Extract features and classify the current contour.
        gesture_res = self.gesture_recognizer.process_gesture(
            contour=hand_res.selected_contour,
            contour_score=hand_res.score,
            history_confidence=0.0,
        )

        # Step 4: Update history with the raw classification before measuring
        # temporal confidence for this frame.
        raw_gesture = gesture_res.raw_gesture if hand_res.is_hand_detected else None
        history_res = self.gesture_history.process_history(raw_gesture)

        # Step 5: Recalculate confidence from the current raw gesture and the
        # history state that already includes this frame.
        updated_history_confidence = self.gesture_history.get_confidence(raw_gesture)
        gesture_res.confidence = self.gesture_recognizer.calculate_confidence(
            contour=hand_res.selected_contour,
            contour_score=hand_res.score,
            features=gesture_res.features,
            gesture_label=gesture_res.raw_gesture,
            history_confidence=updated_history_confidence,
        )

        return DetectionFrameResult(
            frame=frame,
            fps=fps,
            skin_res=skin_res,
            hand_res=hand_res,
            gesture_res=gesture_res,
            history_res=history_res,
        )

    def run(self) -> None:
        """Main event loop."""
        if not self.calibrated:
            print("Not calibrated. Exiting.")
            return

        try:
            while True:
                diag = self.process_current_frame()
                if diag is None:
                    print("Camera dropped. Exiting.")
                    break

                output_frame = diag.frame.copy()

                if diag.hand_res.is_hand_detected and diag.hand_res.selected_contour is not None:
                    output_frame = self.hand_detector.draw_both(output_frame, diag.hand_res.selected_contour)

                    smoothed = diag.history_res.smoothed_gesture
                    action = diag.history_res.action
                    confidence = diag.gesture_res.confidence

                    y_offset = 40
                    if smoothed == "Unknown":
                        cv2.putText(
                            output_frame,
                            f"Gesture: Unknown | Confidence: {confidence:.0f}%",
                            (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 165, 255),
                            2,
                        )
                    elif smoothed:
                        cv2.putText(
                            output_frame,
                            f"Gesture: {smoothed} ({confidence:.0f}%)",
                            (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            output_frame,
                            f"Action: {action}",
                            (20, y_offset + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 255),
                            2,
                        )
                    else:
                        cv2.putText(
                            output_frame,
                            f"Raw: {diag.gesture_res.raw_gesture} ({confidence:.0f}%)",
                            (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (100, 100, 255),
                            2,
                        )
                else:
                    cv2.putText(
                        output_frame,
                        f"No Hand Detected | Confidence: {diag.gesture_res.confidence:.0f}%",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                cv2.putText(
                    output_frame,
                    f"FPS: {diag.fps:.1f}",
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
                    try:
                        cv2.destroyWindow("Hand Gesture Recognition")
                    except Exception:
                        pass
                    print("\nRECALIBRATION REQUESTED")
                    print("-" * 50)
                    recal_res = self._run_calibration_interactive()

                    if recal_res.success and recal_res.lower_hsv is not None and recal_res.upper_hsv is not None:
                        self.skin_detector = SkinDetector(recal_res.lower_hsv, recal_res.upper_hsv)
                        CalibrationManager.save_calibration(recal_res.lower_hsv, recal_res.upper_hsv)
                        self.calibrated = True
                        print("✓ Recalibration Successful!")
                    else:
                        if self.calibrated:
                            print(f"⚠ Recalibration {recal_res.status.lower()}. Previous Calibration Retained.")
                        else:
                            print("✗ Recalibration Failed. No valid calibration exists.")

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
