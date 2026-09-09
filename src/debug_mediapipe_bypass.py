"""
Standalone MediaPipe Hand & Landmark Detection Diagnostic Tool.

Implements the 5-step diagnostic procedure to isolate MediaPipe hand detection:
1. Confirm MediaPipe detects a hand and draws the 21 landmarks.
2. Check confidence thresholds (supports 0.5 and 0.3 permissive values).
3. Verify BGR -> RGB color space conversion.
4. Verify extraction of all 21 keypoints.
5. Bypass gesture classification logic entirely, printing:
     HAND DETECTED
     Landmarks: 21
   or:
     NO HAND
"""

import argparse
import os
import sys
import cv2

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.camera import CameraCapture
from src.mediapipe_detector import MediaPipeDetector


def run_bypass_diagnostic(
    camera_id: int = 0,
    min_confidence: float = 0.5,
    draw_labels: bool = True,
    width: int = 640,
    height: int = 480,
    auto_contrast: bool = False,
) -> None:
    print("=" * 68)
    print("  MediaPipe 3D Landmark Isolation & Diagnostic Test")
    print("=" * 68)
    print(f"Camera ID:              {camera_id}")
    print(f"Target Resolution:      {width}x{height}")
    print(f"Detection Confidence:   {min_confidence}")
    print(f"Tracking Confidence:    {min_confidence}")
    print(f"Draw Landmark IDs:      {draw_labels}")
    print(f"CLAHE Auto Contrast:    {auto_contrast}")
    print("=" * 68)

    try:
        camera = CameraCapture(
            camera_id=camera_id,
            target_width=width,
            target_height=height,
            auto_contrast=auto_contrast,
        )
        print("✓ Camera initialized successfully")
    except Exception as e:
        print(f"✗ Camera initialization error: {e}")
        sys.exit(1)

    try:
        detector = MediaPipeDetector(
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            max_num_hands=1,
            static_image_mode=False,
        )
        print("✓ MediaPipe HandLandmarker initialized successfully")
    except Exception as e:
        print(f"✗ MediaPipe detector error: {e}")
        camera.release()
        sys.exit(1)

    print("\nStarting video stream. Press 'q' in the window or Ctrl+C to quit.\n")

    frame_idx = 0
    try:
        while True:
            ret, frame = camera.get_frame()
            if not ret or frame is None:
                print("Camera feed dropped. Exiting.")
                break

            frame_idx += 1

            # Process frame through MediaPipe
            landmarks, annotated_frame = detector.process_frame(
                frame, draw_labels=draw_labels
            )

            # Core diagnostic test: Print HAND DETECTED / NO HAND
            if landmarks is not None:
                print("HAND DETECTED")
                print(f"Landmarks: {len(landmarks)}")
            else:
                print("NO HAND")

            # Diagnostic HUD
            h, w = annotated_frame.shape[:2]
            status_text = "HAND DETECTED" if landmarks is not None else "NO HAND"
            status_color = (0, 255, 0) if landmarks is not None else (0, 0, 255)
            lm_count = len(landmarks) if landmarks is not None else 0

            cv2.putText(
                annotated_frame,
                f"MediaPipe Status: {status_text}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Landmarks Count: {lm_count} / 21",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Threshold: {min_confidence:.2f} | FPS: {camera.get_fps():.1f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )
            cv2.putText(
                annotated_frame,
                "Press 'q' to exit",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 180),
                1,
            )

            cv2.imshow("MediaPipe Detection Diagnostic", annotated_frame)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nDiagnostic interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        detector.close()
        camera.release()
        print("✓ Diagnostic complete. Resources cleanly released.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MediaPipe Landmark Detection Diagnostic & Classifier Bypass"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detection and tracking (default: 0.5; try 0.3 for low light).",
    )
    parser.add_argument(
        "--permissive",
        action="store_true",
        help="Shortcut for --confidence 0.3 (permissive debugging mode).",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width (default: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height (default: 480).",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not draw numerical keypoint IDs on skeleton joints.",
    )
    parser.add_argument(
        "--auto-contrast",
        action="store_true",
        help="Enable adaptive CLAHE contrast enhancement for dim or backlit rooms.",
    )

    args = parser.parse_args()
    conf = 0.3 if args.permissive else args.confidence

    run_bypass_diagnostic(
        camera_id=args.camera_id,
        min_confidence=conf,
        draw_labels=not args.no_labels,
        width=args.width,
        height=args.height,
        auto_contrast=args.auto_contrast,
    )


if __name__ == "__main__":
    main()
