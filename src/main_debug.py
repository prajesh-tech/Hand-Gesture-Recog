"""
Debug version of main.py that displays comprehensive real-time diagnostics overlay.
Shows raw/clean skin pixel counts, percentages, HSV bounds, contour scores, gesture, and confidence %.
"""

import os
import sys

import cv2

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.calibration import CalibrationManager
from src.camera import CameraCapture
from src.gesture_history import GestureHistory
from src.gesture_recognition import GestureRecognizer
from src.hand_detection import HandDetector
from src.skin_detection import SkinDetector


def debug_run() -> None:
    """Main debug loop."""
    frame_w, frame_h = 640, 480
    camera = CameraCapture(target_width=frame_w, target_height=frame_h)
    print("✓ Camera initialized")

    calibration = CalibrationManager.load_calibration()
    if calibration:
        lower, upper = calibration
        print(f"✓ Loaded calibration: Lower={list(lower)}, Upper={list(upper)}")
    else:
        print("⚠ No saved calibration found. Using default HSV thresholds.")
        lower, upper = SkinDetector.DEFAULT_LOWER_HSV.copy(), SkinDetector.DEFAULT_UPPER_HSV.copy()

    skin_detector = SkinDetector(lower, upper)
    hand_detector = HandDetector(
        min_contour_area=500,
        frame_width=frame_w,
        frame_height=frame_h,
    )
    gesture_recognizer = GestureRecognizer()
    gesture_history = GestureHistory(buffer_size=8, consensus_threshold=4)

    print("\n📊 DEBUG MODE - Real-time Analysis")
    print("=" * 60)
    print("Press 'q' to quit")
    print("=" * 60)

    frame_count = 0

    try:
        while True:
            ret, frame = camera.get_frame()
            if not ret or frame is None:
                print("✗ Camera error")
                break

            frame_count += 1
            h, w = frame.shape[:2]

            # 1. Skin detection
            skin_res = skin_detector.process_frame(frame, morphology_op="both")

            # 2. Hand detection
            hand_res = hand_detector.process_mask(skin_res.mask_clean)

            # 3. Extract features and classify the current contour.
            gesture_res = gesture_recognizer.process_gesture(
                contour=hand_res.selected_contour,
                contour_score=hand_res.score,
                history_confidence=0.0,
            )

            # 4. Update temporal history before calculating confidence.
            raw_label = gesture_res.raw_gesture if hand_res.is_hand_detected else None
            history_res = gesture_history.process_history(raw_label)
            updated_history_confidence = gesture_history.get_confidence(raw_label)
            gesture_res.confidence = gesture_recognizer.calculate_confidence(
                contour=hand_res.selected_contour,
                contour_score=hand_res.score,
                features=gesture_res.features,
                gesture_label=gesture_res.raw_gesture,
                history_confidence=updated_history_confidence,
            )

            # 5. Render overlay
            output = frame.copy()
            text_y = 25

            fps = camera.get_fps()
            cv2.putText(
                output,
                f"Resolution: {w}x{h} | FPS: {fps:.1f}",
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
            text_y += 22

            cv2.putText(
                output,
                f"HSV Lower: {list(lower)} | Upper: {list(upper)}",
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 255),
                1,
            )
            text_y += 22

            cv2.putText(
                output,
                f"Raw Skin: {skin_res.raw_pixel_count} px ({skin_res.raw_percentage:.1f}%)",
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
            )
            text_y += 22

            cv2.putText(
                output,
                f"Clean Skin: {skin_res.clean_pixel_count} px ({skin_res.clean_percentage:.1f}%)",
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
            )
            text_y += 25

            total_cnt = len(hand_res.candidates) + len(hand_res.rejected)
            cv2.putText(
                output,
                f"Contours: {total_cnt} | Candidates: {len(hand_res.candidates)}",
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 200, 0),
                1,
            )
            text_y += 25

            if hand_res.is_hand_detected and hand_res.selected_contour is not None:
                cv2.putText(
                    output,
                    f"Selected Area: {hand_res.area:.0f} px | Contour Score: {hand_res.score:.2f}",
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                )
                text_y += 25

                output = hand_detector.draw_both(output, hand_res.selected_contour)
            else:
                cv2.putText(
                    output,
                    f"Gesture: No Hand Detected | Confidence: {gesture_res.confidence:.0f}%",
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                )

            features = gesture_res.features or {}
            raw_text = gesture_res.raw_gesture or "None"
            stable_text = history_res.smoothed_gesture or "None"
            history_text = ", ".join("None" if item is None else item for item in history_res.history)
            consensus_label = history_res.smoothed_gesture or gesture_res.raw_gesture
            consensus_count = (
                sum(item == consensus_label for item in history_res.history)
                if consensus_label is not None
                else 0
            )
            consensus_text = f"{consensus_count}/{len(history_res.history)}"

            debug_lines = [
                f"Raw Gesture: {raw_text}",
                f"Stable Gesture: {stable_text}",
                f"Defects: {features.get('convexity_defects_count', 0)} | Solidity: {features.get('solidity', 0.0):.3f}",
                f"Elongation: {features.get('elongation', 0.0):.3f} | Extent: {features.get('extent', 0.0):.3f}",
                f"History: [{history_text}] | Consensus: {consensus_text}",
                f"Confidence: {gesture_res.confidence:.1f}% | Temporal: {updated_history_confidence:.2f}",
            ]
            for line in debug_lines:
                cv2.putText(
                    output,
                    line,
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                text_y += 22

            # Draw ROI overlay for calibration reference
            roi_frame = frame.copy()
            cy, cx = h // 2, w // 2
            rh, rw = int(h * 0.25), int(w * 0.25)
            cv2.rectangle(roi_frame, (cx - rw, cy - rh), (cx + rw, cy + rh), (0, 255, 0), 2)

            # Show debug windows
            cv2.imshow("Main (with stats)", output)
            cv2.imshow("Skin Mask (raw)", skin_res.mask_raw)
            cv2.imshow("Skin Mask (cleaned)", skin_res.mask_clean)
            cv2.imshow("Calibration ROI", roi_frame)

            if frame_count % 30 == 0:
                print(
                    f"[Frame {frame_count}] Cleaned skin: {skin_res.clean_percentage:.1f}%, "
                    f"Gesture: {gesture_res.gesture}, Confidence: {gesture_res.confidence:.0f}%"
                )

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        camera.release()
        print("✓ Debug session ended")


if __name__ == "__main__":
    debug_run()
