"""
Main application: Real-time hand gesture recognition with HSV-based skin detection.
Orchestrates camera, preprocessing, skin detection, hand detection, and gesture recognition.
"""

import cv2
import numpy as np
import sys
import os
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.camera import CameraCapture
from src.skin_detection import SkinDetector
from src.hand_detection import HandDetector
from src.gesture_recognition import GestureRecognizer
from src.gesture_history import GestureHistory
from src.calibration import CalibrationManager


class HandGestureApp:
    """Main application class orchestrating all components."""
    
    # Gesture to action mapping
    GESTURE_ACTIONS = {
        "Fist": "STOP",
        "Open Palm": "START",
        "One Finger": "SELECT",
        "Two Fingers": "NEXT",
    }
    
    def __init__(self, camera_id: int = 0, frame_width: int = 640, frame_height: int = 480,
                 resize_factor: float = 1.0, skip_calibration: bool = False):
        """
        Initialize the application.
        
        Args:
            camera_id: Webcam ID
            frame_width: Target frame width
            frame_height: Target frame height
            resize_factor: Frame resize factor for performance (0.5 = half size)
            skip_calibration: Skip calibration even if not saved (for testing)
        """
        print("🎬 Hand Gesture Recognition System")
        print("=" * 50)
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.resize_factor = resize_factor
        
        # Initialize modules
        try:
            self.camera = CameraCapture(
                camera_id=camera_id,
                target_width=frame_width,
                target_height=frame_height,
                resize_factor=resize_factor
            )
            print("✓ Camera initialized")
        except RuntimeError as e:
            print(f"✗ Camera error: {str(e)}")
            sys.exit(1)
        
        self.skin_detector = SkinDetector()
        print("✓ Skin detector initialized")
        
        self.hand_detector = HandDetector(min_contour_area=500)
        print("✓ Hand detector initialized")
        
        self.gesture_recognizer = GestureRecognizer()
        print("✓ Gesture recognizer initialized")
        
        self.gesture_history = GestureHistory(buffer_size=10, consensus_threshold=5)
        print("✓ Gesture history (temporal smoothing) initialized")
        
        # Calibration
        self.skip_calibration = skip_calibration
        self.calibrated = False
        
        # Run calibration if needed
        if not skip_calibration:
            self._handle_calibration()
        else:
            # Use default HSV thresholds
            self.calibrated = True
            print("⚠ Calibration skipped (using defaults)")
        
        print("=" * 50)
        print("Ready to run. Press 'q' to quit, 'c' to recalibrate.")
        print("=" * 50)
    
    def _handle_calibration(self) -> None:
        """Check for saved calibration, or run interactive calibration."""
        # Try loading saved calibration
        calibration = CalibrationManager.load_calibration()
        
        if calibration:
            lower, upper = calibration
            self.skin_detector.set_hsv_range(lower, upper)
            self.calibrated = True
            print("✓ Loaded saved calibration")
            return
        
        # Run interactive calibration
        print("\n🎯 CALIBRATION MODE")
        print("-" * 50)
        self._run_calibration_interactive()
    
    def _run_calibration_interactive(self) -> None:
        """
        Interactive calibration: user places hand, presses SPACE to sample.
        Computes HSV range from samples and saves.
        """
        print("Instructions:")
        print("1. Place your PALM in the center of the frame")
        print("2. Press SPACE to sample (capture 10 samples)")
        print("3. Keep hand steady during sampling")
        print("4. Calibration will auto-save")
        print("\nPress ESC to skip calibration (use defaults)")
        print("-" * 50)
        
        samples = []
        sample_count = 0
        target_samples = 10
        
        while True:
            ret, frame = self.camera.get_frame()
            if not ret:
                print("✗ Camera error during calibration")
                self.calibrated = False
                return
            
            # Draw instructions on frame
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 3)
            cv2.putText(frame, f"Samples: {sample_count}/{target_samples}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if sample_count > 0:
                cv2.putText(frame, "Sampling...", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Press SPACE to start", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            cv2.imshow("Calibration", frame)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord(' '):  # SPACE key
                # Sample from center region
                center_y, center_x = h // 2, w // 2
                region_size = 60
                y1 = max(0, center_y - region_size)
                y2 = min(h, center_y + region_size)
                x1 = max(0, center_x - region_size)
                x2 = min(w, center_x + region_size)
                
                sample_region = frame[y1:y2, x1:x2]
                samples.append(sample_region)
                sample_count += 1
                
                if sample_count >= target_samples:
                    break
            
            elif key == 27:  # ESC key
                print("⚠ Calibration skipped")
                self.calibrated = False
                cv2.destroyWindow("Calibration")
                return
        
        cv2.destroyWindow("Calibration")
        
        # Compute HSV range from all samples
        print("\nProcessing samples...")
        combined_sample = np.vstack(samples)
        lower, upper = SkinDetector.extract_hsv_from_region(
            combined_sample,
            np.ones(combined_sample.shape[:2], dtype=np.uint8) * 255  # Mask is all white
        )
        
        self.skin_detector.set_hsv_range(lower, upper)
        
        # Save calibration
        CalibrationManager.save_calibration(lower, upper)
        self.calibrated = True
        
        print(f"✓ Calibration complete!")
        print(f"  Lower HSV: {lower}")
        print(f"  Upper HSV: {upper}")
    
    def run(self) -> None:
        """Main event loop."""
        if not self.calibrated:
            print("✗ Not calibrated. Exiting.")
            return
        
        try:
            while True:
                ret, frame = self.camera.get_frame()
                if not ret:
                    print("✗ Camera dropped. Exiting.")
                    break
                
                # Preprocessing
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
                
                # Skin detection
                skin_mask = self.skin_detector.detect_and_clean(frame_blurred, morphology_op='both')
                
                # Hand detection
                hand_contour = self.hand_detector.find_hand_contour(skin_mask)
                
                # Process output frame
                output_frame = frame.copy()
                
                if hand_contour is not None:
                    # Draw hand contour and bounding box
                    output_frame = self.hand_detector.draw_both(output_frame, hand_contour)
                    
                    # Recognize gesture
                    gesture = self.gesture_recognizer.recognize_gesture(hand_contour)
                    self.gesture_history.add_frame(gesture)
                    
                    # Get smoothed gesture (with consensus)
                    smoothed_gesture = self.gesture_history.get_smoothed_gesture()
                    
                    # Get action
                    action = self.GESTURE_ACTIONS.get(smoothed_gesture, "")
                    
                    # Draw gesture and action on frame
                    y_offset = 40
                    if smoothed_gesture:
                        cv2.putText(output_frame, f"Gesture: {smoothed_gesture}", (20, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        y_offset += 40
                        cv2.putText(output_frame, f"Action: {action}", (20, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:
                        cv2.putText(output_frame, "Gesture: (detecting...)", (20, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
                else:
                    # No hand detected
                    self.gesture_history.add_frame(None)
                    cv2.putText(output_frame, "No Hand Detected", (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Draw FPS
                fps = self.camera.get_fps()
                cv2.putText(output_frame, f"FPS: {fps:.1f}", (20, output_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Display
                cv2.imshow("Hand Gesture Recognition", output_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(30) & 0xFF
                
                if key == ord('q'):  # Quit
                    print("\n👋 Exiting...")
                    break
                
                elif key == ord('c'):  # Recalibrate
                    print("\n🔄 Recalibrating...")
                    cv2.destroyWindow("Hand Gesture Recognition")
                    self._run_calibration_interactive()
                    print("Resuming main loop...")
        
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        cv2.destroyAllWindows()
        self.camera.release()
        print("✓ Cleanup complete")


def main():
    """Entry point."""
    app = HandGestureApp(
        camera_id=0,
        frame_width=640,
        frame_height=480,
        resize_factor=1.0,
        skip_calibration=False
    )
    app.run()


if __name__ == "__main__":
    main()
