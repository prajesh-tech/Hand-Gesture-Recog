"""
Debug version of main.py that displays comprehensive real-time diagnostics.
Shows raw/clean skin pixel counts, percentages, HSV bounds, contour counts, and metrics.
"""

import os
import sys

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.calibration import CalibrationManager
from src.camera import CameraCapture
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
        skin_model = CalibrationManager.load_skin_model()
        print(f"✓ Loaded calibration: Lower={list(lower)}, Upper={list(upper)}")
    else:
        print("⚠ No saved calibration found. Using default HSV thresholds.")
        lower, upper = SkinDetector.DEFAULT_LOWER_HSV.copy(), SkinDetector.DEFAULT_UPPER_HSV.copy()
        skin_model = None

    skin_detector = SkinDetector(lower, upper, statistical_model=skin_model)
    hand_detector = HandDetector(
        min_contour_area=500,
        frame_width=frame_w,
        frame_height=frame_h,
    )

    print("\n📊 DEBUG MODE - Real-time Analysis")
    print("=" * 60)
    print("Press 'q' to quit")
    print("=" * 60)

    frame_count = 0

    while True:
        ret, frame = camera.get_frame()
        if not ret:
            print("✗ Camera error")
            break

        frame_count += 1
        h, w = frame.shape[:2]
        total_pixels = float(h * w)

        # 1. Skin detection
        skin_mask_raw = skin_detector.detect_skin(frame)
        skin_mask_clean = skin_detector.apply_morphology(skin_mask_raw, operation="both")

        # 2. Pixel diagnostics
        raw_pixels = int(np.count_nonzero(skin_mask_raw))
        clean_pixels = int(np.count_nonzero(skin_mask_clean))
        raw_pct = (100.0 * raw_pixels) / total_pixels
        clean_pct = (100.0 * clean_pixels) / total_pixels

        # 3. Hand detection & diagnostics
        diag = hand_detector.get_diagnostic_info(skin_mask_clean)
        selected_contour = diag.get("selected")
        cand_count = diag.get("candidate_count", 0)
        total_contours = diag.get("total_contours", 0)

        # 4. Render overlay
        output = frame.copy()
        text_y = 25

        fps = camera.get_fps()
        cv2.putText(output, f"Resolution: {w}x{h} | FPS: {fps:.1f}", (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        text_y += 22

        cv2.putText(output, f"HSV Lower: {list(lower)} | Upper: {list(upper)}", (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        text_y += 22

        cv2.putText(output, f"Raw Skin: {raw_pixels} px ({raw_pct:.1f}%)", (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        text_y += 22

        cv2.putText(output, f"Cleaned Skin: {clean_pixels} px ({clean_pct:.1f}%)", (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        text_y += 25

        cv2.putText(output, f"Contours Total: {total_contours} | Candidates: {cand_count}", (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)
        text_y += 25

        if selected_contour is not None:
            area = diag["selected_area"]
            perimeter = diag["selected_perimeter"]
            cv2.putText(output, f"✓ HAND DETECTED - Area: {area:.0f} px | Perim: {perimeter:.0f} px",
                        (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            output = hand_detector.draw_both(output, selected_contour)
        else:
            cv2.putText(output, "✗ NO HAND DETECTED", (10, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # Draw ROI overlay for calibration reference
        roi_frame = frame.copy()
        cy, cx = h // 2, w // 2
        rh, rw = int(h * 0.25), int(w * 0.25)
        cv2.rectangle(roi_frame, (cx - rw, cy - rh), (cx + rw, cy + rh), (0, 255, 0), 2)

        # Show debug windows
        cv2.imshow("Main (with stats)", output)
        cv2.imshow("Skin Mask (raw)", skin_mask_raw)
        cv2.imshow("Skin Mask (cleaned)", skin_mask_clean)
        cv2.imshow("Calibration ROI", roi_frame)

        if frame_count % 30 == 0:
            print(f"[Frame {frame_count}] Raw skin: {raw_pct:.1f}%, Cleaned: {clean_pct:.1f}%, Candidates: {cand_count}")

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    camera.release()
    print("✓ Debug session ended")


if __name__ == "__main__":
    debug_run()
