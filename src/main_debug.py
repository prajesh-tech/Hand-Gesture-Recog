"""
Debug version of main.py that prints detailed information.
Run this to see what's happening with calibration and hand detection.
"""

import os
import sys

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.calibration import CalibrationManager
from src.camera import CameraCapture
from src.hand_detection import HandDetector
from src.skin_detection import SkinDetector


def debug_run():
    """Main debug loop."""
    # Initialize camera
    camera = CameraCapture(target_width=640, target_height=480)
    print("✓ Camera initialized")
    
    # Load or calibrate
    calibration = CalibrationManager.load_calibration()
    if calibration:
        lower, upper = calibration
        print(f"✓ Loaded calibration: Lower={lower}, Upper={upper}")
    else:
        print("✗ No calibration found. Please delete config/hsv_calibration.json and run main.py first")
        return
    
    skin_detector = SkinDetector()
    skin_detector.set_hsv_range(lower, upper)
    hand_detector = HandDetector(min_contour_area=500)
    
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
        
        # 1. Convert to HSV
        frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        frame_hsv = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)  # noqa: F841
        
        # 2. Skin detection
        skin_mask = skin_detector.detect_skin(frame_blurred)
        skin_mask_clean = skin_detector.apply_morphology(skin_mask, operation='both')
        
        # 3. Count pixels
        skin_pixels = np.count_nonzero(skin_mask)
        skin_pixels_clean = np.count_nonzero(skin_mask_clean)
        
        # 4. Hand detection
        hand_contour = hand_detector.find_hand_contour(skin_mask_clean)
        
        # 5. Create output
        output = frame.copy()
        
        # Show statistics
        text_y = 30
        cv2.putText(output, f"Frame: {frame_count}", (10, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        text_y += 25
        
        cv2.putText(output, f"Lower HSV: {list(lower)}", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        text_y += 20
        
        cv2.putText(output, f"Upper HSV: {list(upper)}", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        text_y += 25
        
        cv2.putText(output, f"Skin pixels (raw): {skin_pixels}", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        text_y += 25
        
        cv2.putText(output, f"Skin pixels (clean): {skin_pixels_clean}", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        text_y += 30
        
        if hand_contour is not None:
            area = cv2.contourArea(hand_contour)
            perimeter = cv2.arcLength(hand_contour, True)
            cv2.putText(output, f"✓ HAND DETECTED - Area: {area:.0f}", (10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            text_y += 25
            cv2.putText(output, f"  Perimeter: {perimeter:.0f}", (10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 1)
            
            # Draw the hand
            output = hand_detector.draw_both(output, hand_contour)
        else:
            cv2.putText(output, "✗ NO HAND DETECTED", (10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show images
        cv2.imshow("Main (with stats)", output)
        cv2.imshow("Skin Mask (raw)", skin_mask)
        cv2.imshow("Skin Mask (cleaned)", skin_mask_clean)
        
        # Print to console periodically
        if frame_count % 30 == 0:
            print(f"\n[Frame {frame_count}]")
            print(f"  Skin pixels (raw): {skin_pixels}")
            print(f"  Skin pixels (clean): {skin_pixels_clean}")
            if hand_contour is not None:
                print(f"  Hand detected: YES (area={area:.0f})")
            else:
                print("  Hand detected: NO")
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("\n✓ Debug session ended")


if __name__ == "__main__":
    debug_run()
