"""
Improved main application with robust calibration and diagnostics.
Uses enhanced skin detection and hand detection.
"""

import cv2
import numpy as np
import sys
import os
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.skin_detection_improved import SkinDetectorImproved
from src.hand_detection_improved import HandDetectorImproved
from src.gesture_recognition import GestureRecognizer
from src.gesture_history import GestureHistory
from src.calibration import CalibrationManager
from src.camera import CameraCapture


class HandGestureAppImproved:
    """Improved hand gesture app with robust calibration and diagnostics."""
    
    GESTURE_ACTIONS = {
        "Fist": "STOP",
        "Open Palm": "START",
        "One Finger": "SELECT",
        "Two Fingers": "NEXT",
    }
    
    def __init__(self, camera_id: int = 0, frame_width: int = 640, frame_height: int = 480,
                 resize_factor: float = 1.0, skip_calibration: bool = False,
                 morphology_strength: str = 'normal'):
        """
        Initialize improved app.
        
        Args:
            morphology_strength: 'light', 'normal', or 'strong'
        """
        print("🎬 Hand Gesture Recognition System (IMPROVED)")
        print("=" * 60)
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.resize_factor = resize_factor
        
        # Initialize camera
        try:
            self.camera = CameraCapture(
                camera_id=camera_id,
                target_width=frame_width,
                target_height=frame_height,
                resize_factor=resize_factor
            )
            print("✓ Camera initialized")
        except RuntimeError as e:
            print(f"✗ Camera error: {e}")
            sys.exit(1)
        
        # Initialize improved skin detector
        self.skin_detector = SkinDetectorImproved(
            morphology_kernel_size=5,
            morphology_strength=morphology_strength
        )
        print("✓ Improved skin detector initialized")
        
        # Initialize improved hand detector
        self.hand_detector = HandDetectorImproved(
            min_contour_area=500,
            max_contour_area=0.7 * frame_width * frame_height,
            frame_width=frame_width,
            frame_height=frame_height
        )
        print("✓ Improved hand detector initialized")
        
        # Initialize gesture recognizer
        self.gesture_recognizer = GestureRecognizer()
        print("✓ Gesture recognizer initialized")
        
        # Initialize gesture history
        self.gesture_history = GestureHistory(buffer_size=10, consensus_threshold=5)
        print("✓ Gesture history initialized")
        
        self.skip_calibration = skip_calibration
        self.calibrated = False
        self.last_calibration_stats = {}
        
        # Run calibration if needed
        if not skip_calibration:
            self._handle_calibration()
        else:
            self.calibrated = True
            print("⚠ Calibration skipped (using defaults)")
        
        print("=" * 60)
        print("Ready to run. Press 'q' to quit, 'c' to recalibrate")
        print("=" * 60)
    
    def _handle_calibration(self) -> None:
        """Check for saved calibration or run interactive calibration."""
        calibration = CalibrationManager.load_calibration()
        
        if calibration:
            lower, upper = calibration
            self.skin_detector.set_hsv_range(lower, upper)
            self.calibrated = True
            print("✓ Loaded saved calibration")
            return
        
        print("\n🎯 CALIBRATION MODE (IMPROVED)")
        print("-" * 60)
        self._run_calibration_interactive()
    
    def _run_calibration_interactive(self) -> None:
        """
        Interactive calibration with robust sampling and validation.
        """
        print("Instructions:")
        print("1. Place your PALM flat in the center of the frame")
        print("2. Ensure GOOD LIGHTING (no shadows)")
        print("3. Press SPACE to capture sample (need 10 samples)")
        print("4. Keep hand STEADY during sampling")
        print("5. System will validate and save calibration")
        print("\nPress ESC to skip calibration")
        print("-" * 60)
        
        samples = []
        sample_count = 0
        target_samples = 10
        
        while True:
            ret, frame = self.camera.get_frame()
            if not ret:
                print("✗ Camera error during calibration")
                self.calibrated = False
                return
            
            # Draw UI on frame
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 3)
            
            cv2.putText(frame, f"Samples: {sample_count}/{target_samples}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            if sample_count == 0:
                cv2.putText(frame, "Press SPACE to start sampling", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                cv2.putText(frame, f"Keep hand steady... {sample_count} captured", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("Calibration", frame)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord(' '):  # SPACE
                samples.append(frame.copy())
                sample_count += 1
                print(f"  Sample {sample_count}/{target_samples} captured")
                
                if sample_count >= target_samples:
                    break
            
            elif key == 27:  # ESC
                print("⚠ Calibration skipped")
                self.calibrated = False
                cv2.destroyWindow("Calibration")
                return
        
        cv2.destroyWindow("Calibration")
        
        # Compute calibration from samples
        print("\nProcessing samples...")
        lower, upper, stats = SkinDetectorImproved.calibrate_from_samples(samples)
        
        print("\n📊 Calibration Results:")
        print(f"  Total samples: {stats.get('total_samples', 'N/A')}")
        print(f"  Total pixels analyzed: {stats.get('total_pixels', 'N/A')}")
        print(f"  Valid pixels (S>20, V>40): {stats.get('valid_pixels', 'N/A')}")
        print(f"  Valid percentage: {stats.get('valid_percentage', 'N/A'):.1f}%")
        print(f"  Hue wrap-around detected: {stats.get('hue_wrap_detected', False)}")
        print(f"  Hue range: {stats.get('hue_range', 'N/A')}")
        print(f"  Saturation range: {stats.get('sat_range', 'N/A')}")
        print(f"  Value range: {stats.get('val_range', 'N/A')}")
        
        if 'warning' in stats:
            print(f"  ⚠ {stats['warning']}")
        
        # Validate calibration
        if stats.get('valid_percentage', 0) < 20:
            print("\n✗ CALIBRATION FAILED: Less than 20% valid pixels")
            print("  Likely causes:")
            print("  - Hand not clearly visible in center")
            print("  - Poor lighting conditions")
            print("  - High background contamination")
            print("\nPlease recalibrate with better conditions.")
            self.calibrated = False
            return
        
        print(f"\nFinal HSV bounds:")
        print(f"  Lower: {list(lower)}")
        print(f"  Upper: {list(upper)}")
        
        # Apply and save
        self.skin_detector.set_hsv_range(lower, upper)
        CalibrationManager.save_calibration(lower, upper)
        self.last_calibration_stats = stats
        self.calibrated = True
        
        print("✓ Calibration complete and saved!")
    
    def run(self) -> None:
        """Main event loop with diagnostics."""
        if not self.calibrated:
            print("✗ Not calibrated. Exiting.")
            return
        
        print("\n🎮 Starting gesture recognition...")
        print("Press 'q' to quit, 'c' to recalibrate, 'd' for diagnostics")
        
        frame_count = 0
        show_diagnostics = False
        
        try:
            while True:
                ret, frame = self.camera.get_frame()
                if not ret:
                    print("✗ Camera error. Exiting.")
                    break
                
                frame_count += 1
                
                # Preprocessing
                frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
                
                # Skin detection
                skin_mask = self.skin_detector.detect_and_clean(
                    frame_blurred,
                    morphology_op='both',
                    use_statistical=False
                )
                
                # Hand detection
                hand_contour = self.hand_detector.find_hand_contour(skin_mask)
                
                # Compute diagnostics
                h, w = frame.shape[:2]
                frame_area = h * w
                skin_pixels = np.count_nonzero(skin_mask)
                skin_percentage = 100.0 * skin_pixels / frame_area
                
                # Create output frame
                output = frame.copy()
                
                # Draw diagnostics text
                text_y = 30
                lower, upper = self.skin_detector.get_hsv_range()
                
                cv2.putText(output, f"Frame: {frame_count} | Skin: {skin_percentage:.1f}%", 
                           (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                text_y += 25
                
                cv2.putText(output, f"HSV: [{lower[0]},{lower[1]},{lower[2]}] - [{upper[0]},{upper[1]},{upper[2]}]",
                           (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
                text_y += 20
                
                # Hand detection result
                if hand_contour is not None:
                    area = cv2.contourArea(hand_contour)
                    output = self.hand_detector.draw_both(output, hand_contour)
                    
                    cv2.putText(output, f"Hand: {area:.0f} px", (10, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    text_y += 30
                    
                    # Gesture recognition
                    gesture = self.gesture_recognizer.recognize_gesture(hand_contour)
                    self.gesture_history.add_frame(gesture)
                    smoothed_gesture = self.gesture_history.get_smoothed_gesture()
                    
                    if smoothed_gesture:
                        action = self.GESTURE_ACTIONS.get(smoothed_gesture, "")
                        cv2.putText(output, f"Gesture: {smoothed_gesture}", (10, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        text_y += 30
                        cv2.putText(output, f"Action: {action}", (10, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    else:
                        cv2.putText(output, "Gesture: (detecting...)", (10, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 150, 255), 1)
                else:
                    self.gesture_history.add_frame(None)
                    cv2.putText(output, "No Hand Detected", (10, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show main view
                cv2.imshow("Hand Gesture Recognition", output)
                
                # Show masks if diagnostics enabled
                if show_diagnostics or frame_count % 30 == 0:
                    cv2.imshow("Skin Mask (cleaned)", skin_mask)
                
                # Handle keys
                key = cv2.waitKey(30) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    cv2.destroyAllWindows()
                    self._handle_calibration()
                    print("Recalibration complete. Resuming...")
                elif key == ord('d'):
                    show_diagnostics = not show_diagnostics
                    if show_diagnostics:
                        print("📊 Diagnostics enabled")
                    else:
                        print("📊 Diagnostics disabled")
        
        except KeyboardInterrupt:
            print("\n⌛ Interrupted by user")
        
        finally:
            cv2.destroyAllWindows()
            print("✓ Application closed")


def main():
    """Entry point."""
    app = HandGestureAppImproved(
        camera_id=0,
        frame_width=640,
        frame_height=480,
        morphology_strength='normal'
    )
    app.run()


if __name__ == "__main__":
    main()
