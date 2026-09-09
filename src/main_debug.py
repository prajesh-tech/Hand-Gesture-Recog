"""
Real-time diagnostic and telemetry runner for MediaPipe 3D Landmark Strategy 1.
Displays comprehensive live metrics: finger ratios, straightness cosines,
heuristic confidence breakdown (Quality, Rule Match, Stability), and consensus history.
"""

import os
import sys
import cv2

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.camera import CameraCapture
from src.gesture_history import GestureHistory
from src.gesture_recognition import LandmarkGestureRecognizer
from src.mediapipe_detector import MediaPipeDetector


def debug_run() -> None:
    """Main real-time diagnostic execution loop."""
    frame_w, frame_h = 640, 480
    camera = CameraCapture(target_width=frame_w, target_height=frame_h)
    print("✓ Camera initialized successfully")

    detector = MediaPipeDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1,
        static_image_mode=False,
    )
    print("✓ MediaPipe 3D Landmark Detector initialized")

    gesture_recognizer = LandmarkGestureRecognizer()
    gesture_history = GestureHistory(buffer_size=8, consensus_threshold=4)

    print("\n📊 STRATEGY 1 DEBUG MODE - Real-Time 3D Landmark Geometry")
    print("=" * 70)
    print("Press 'q' in video window or Ctrl+C to quit")
    print("=" * 70)

    frame_count = 0

    try:
        while True:
            ret, frame = camera.get_frame()
            if not ret or frame is None:
                print("✗ Camera read failure. Exiting.")
                break

            frame_count += 1
            fps = camera.get_fps()

            # 1. Landmark detection and skeleton visualization
            landmarks, annotated = detector.process_frame(frame)
            is_hand_detected = landmarks is not None

            # 2. Geometric feature extraction and raw classification
            gesture_res = gesture_recognizer.process_gesture(
                landmarks=landmarks,
                history_confidence=0.0,
            )

            # 3. Temporal consensus tracking
            raw_label = gesture_res.raw_gesture if is_hand_detected else None
            history_res = gesture_history.process_history(raw_label)

            # 4. Confidence recalculation with history consensus
            updated_hist_conf = gesture_history.get_confidence(raw_label)
            confidence, breakdown = gesture_recognizer.calculate_confidence(
                landmarks=landmarks,
                features=gesture_res.features,
                gesture_label=raw_label,
                history_confidence=updated_hist_conf,
            )
            gesture_res.confidence = confidence
            gesture_res.confidence_breakdown = breakdown

            # 5. Diagnostic HUD rendering
            output = annotated.copy()
            text_y = 25

            # System telemetry
            cv2.putText(
                output,
                f"Resolution: {frame_w}x{frame_h} | FPS: {fps:.1f}",
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
            text_y += 24

            if is_hand_detected and gesture_res.features is not None:
                feat = gesture_res.features
                cv2.putText(
                    output,
                    f"Hand: Detected | Palm Size: {feat['palm_size']:.3f} | Var: {feat['coord_variance']:.5f}",
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                )
                text_y += 24

                # Finger geometric states
                thumb = feat["thumb"]
                idx = feat["index"]
                mid = feat["middle"]
                ring = feat["ring"]
                pnk = feat["pinky"]

                thumb_state = "EXTENDED" if thumb["is_extended"] else "TUCKED"
                cv2.putText(
                    output,
                    f"Thumb: {thumb_state} (spread={thumb['ratio']:.2f}, thr={thumb['ratio_threshold']:.2f})",
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 255, 255) if thumb["is_extended"] else (180, 180, 180),
                    1,
                )
                text_y += 20

                digits = [("Index", idx), ("Middle", mid), ("Ring", ring), ("Pinky", pnk)]
                for name, d in digits:
                    d_state = "EXTENDED" if d["is_extended"] else "CLOSED"
                    color = (0, 255, 0) if d["is_extended"] else (150, 150, 150)
                    cv2.putText(
                        output,
                        f"{name:6}: {d_state} (ratio={d['ratio']:.2f}/{d['ratio_threshold']:.2f}, cos={d['straightness']:.2f}/{d['straightness_threshold']:.2f})",
                        (10, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.46,
                        color,
                        1,
                    )
                    text_y += 20

            else:
                cv2.putText(
                    output,
                    "Hand: NOT DETECTED",
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                text_y += 28

            text_y += 6
            # Gesture & Confidence breakdown
            raw_txt = gesture_res.raw_gesture or "None"
            stable_txt = history_res.smoothed_gesture or "None"
            action_txt = history_res.action or "NONE"

            cv2.putText(
                output,
                f"Raw: {raw_txt}  ->  Stable: {stable_txt} [{action_txt}]",
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
            )
            text_y += 24

            # History buffer display
            hist_items = [str(x) if x is not None else "_" for x in history_res.history]
            hist_str = " ".join(hist_items)
            cv2.putText(
                output,
                f"History [{len(history_res.history)}/8]: [{hist_str}]",
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (200, 200, 255),
                1,
            )
            text_y += 22

            # Confidence components
            q = breakdown.get("quality", 0.0)
            rm = breakdown.get("rule_match", 0.0)
            st = breakdown.get("stability", 0.0)
            cv2.putText(
                output,
                f"Confidence: {confidence:.0f}%  [Quality: {q:.1f}/30 | Rule: {rm:.1f}/50 | Stability: {st:.1f}/20]",
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            cv2.imshow("Strategy 1 Diagnostics", output)

            if frame_count % 30 == 0:
                print(
                    f"[Frame {frame_count:04d}] Gesture: {gesture_res.raw_gesture} -> {history_res.smoothed_gesture}, "
                    f"Conf: {confidence:.0f}% (Q:{q:.1f}, R:{rm:.1f}, S:{st:.1f})"
                )

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nDiagnostic session interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        detector.close()
        camera.release()
        print("✓ Diagnostic session cleanly ended.")


if __name__ == "__main__":
    debug_run()
