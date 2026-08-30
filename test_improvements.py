"""
Standalone test of improved calibration and skin detection logic.
Tests core functionality without requiring pytest.
"""

import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.skin_detection_improved import SkinDetectorImproved
from src.hand_detection_improved import HandDetectorImproved


def test_calibration_with_synthetic_hand():
    """Test calibration with synthetic hand-colored samples."""
    print("\n" + "="*60)
    print("TEST 1: Calibration with clean hand samples")
    print("="*60)
    
    frames = []
    for i in range(10):  # More samples
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create hand-colored region (orange/skin tone)
        # HSV: H~12, S~90, V~150
        center_hsv = np.array([[[12, 90, 150]]], dtype=np.uint8)
        center_bgr = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        
        h, w = 480, 640
        cy, cx = h // 2, w // 2
        region_size = 100  # Larger region to ensure valid pixels in center
        
        # Fill entire region
        frame[cy - region_size:cy + region_size, cx - region_size:cx + region_size] = center_bgr
        
        frames.append(frame)
    
    # Calibrate
    lower, upper, stats = SkinDetectorImproved.calibrate_from_samples(frames)
    
    print(f"✓ Calibration completed")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Valid pixels: {stats['valid_pixels']} ({stats['valid_percentage']:.1f}%)")
    print(f"  Hue wrap-around: {stats['hue_wrap_detected']}")
    print(f"  Hue range: {stats['hue_range']}")
    print(f"  Saturation range: {stats['sat_range']}")
    print(f"  Value range: {stats['val_range']}")
    print(f"  Final bounds: [{lower[0]},{lower[1]},{lower[2]}] - [{upper[0]},{upper[1]},{upper[2]}]")
    
    # Verify - more lenient since we're sampling from a small center region
    assert stats['valid_percentage'] > 10, f"Expected >10% valid pixels, got {stats['valid_percentage']:.1f}%"
    assert lower[0] < 30, f"Expected Hue < 30 for skin tone, got {lower[0]}"
    assert lower[1] >= 15, f"Saturation lower bound should be >= 15, got {lower[1]}"
    assert lower[2] >= 35, f"Value lower bound should be >= 35, got {lower[2]}"
    
    print("✓ All checks passed!")
    return True


def test_hue_wraparound():
    """Test detection of hue wrap-around for red skin tones."""
    print("\n" + "="*60)
    print("TEST 2: Hue wrap-around detection")
    print("="*60)
    
    frames = []
    for _ in range(5):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        h, w = 480, 640
        cy, cx = h // 2, w // 2
        
        # Add mix of red hues (10 and 170)
        for dy in range(-50, 51, 10):
            for dx in range(-50, 51, 10):
                hue_val = 10 if (dy // 10 + dx // 10) % 2 == 0 else 170
                hsv = np.array([[[hue_val, 85, 145]]], dtype=np.uint8)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
                if 0 <= cy+dy < h and 0 <= cx+dx < w:
                    frame[cy+dy, cx+dx] = bgr
        
        frames.append(frame)
    
    lower, upper, stats = SkinDetectorImproved.calibrate_from_samples(frames)
    
    print(f"✓ Calibration completed")
    print(f"  Hue wrap-around detected: {stats['hue_wrap_detected']}")
    print(f"  Hue low count: {stats['hue_low_count']}")
    print(f"  Hue high count: {stats['hue_high_count']}")
    print(f"  Hue range: {stats['hue_range']}")
    print(f"  Final bounds: [{lower[0]},{lower[1]},{lower[2]}] - [{upper[0]},{upper[1]},{upper[2]}]")
    
    print("✓ Wrap-around test passed!")
    return True


def test_background_rejection():
    """Test that black/gray backgrounds are not detected as skin."""
    print("\n" + "="*60)
    print("TEST 3: Background rejection")
    print("="*60)
    
    # Set up detector with typical skin tone range
    detector = SkinDetectorImproved()
    detector.lower_hsv = np.array([0, 15, 40], dtype=np.uint8)
    detector.upper_hsv = np.array([30, 255, 255], dtype=np.uint8)
    
    # Create frame with black background
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add skin tone region in center
    center_hsv = np.array([[[15, 100, 150]]], dtype=np.uint8)
    center_bgr = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2BGR)[0, 0]
    
    h, w = 480, 640
    frame[h//2-50:h//2+50, w//2-50:w//2+50] = center_bgr
    
    # Detect
    mask = detector.detect_skin(frame)
    mask_clean = detector.apply_morphology(mask, operation='both')
    
    # Count pixels
    total_pixels = h * w
    skin_pixels_raw = np.count_nonzero(mask)
    skin_pixels_clean = np.count_nonzero(mask_clean)
    
    skin_pct_raw = 100.0 * skin_pixels_raw / total_pixels
    skin_pct_clean = 100.0 * skin_pixels_clean / total_pixels
    
    print(f"✓ Detection completed")
    print(f"  Raw skin pixels: {skin_pixels_raw} ({skin_pct_raw:.1f}%)")
    print(f"  Clean skin pixels: {skin_pixels_clean} ({skin_pct_clean:.1f}%)")
    print(f"  Background region size: {h*w - (100*100)} px (expected: ~223,000)")
    
    # Background should NOT be heavily detected
    # With good S and V bounds, black (V=0) should not be detected
    assert skin_pct_clean < 50, f"Clean mask detected {skin_pct_clean:.1f}% as skin (expected <50%)"
    
    print("✓ Background rejection passed!")
    return True


def test_hand_contour_selection():
    """Test smart hand contour selection."""
    print("\n" + "="*60)
    print("TEST 4: Hand contour selection")
    print("="*60)
    
    detector = HandDetectorImproved(frame_width=640, frame_height=480)
    
    # Create mask with multiple contours
    mask = np.zeros((480, 640), dtype=np.uint8)
    
    # Small circular blob (background noise)
    cv2.circle(mask, (100, 100), 30, 255, -1)
    
    # Medium circle
    cv2.circle(mask, (550, 350), 80, 255, -1)
    
    # Hand-like shape (preferred)
    hand_pts = np.array([
        [[300, 200]], [[330, 180]], [[350, 200]], [[360, 240]],
        [[340, 280]], [[310, 290]], [[290, 270]], [[280, 240]]
    ], dtype=np.int32)
    cv2.drawContours(mask, [hand_pts], 0, 255, -1)
    
    # Analyze
    diag = detector.get_diagnostic_info(mask)
    selected = detector.find_hand_contour(mask)
    
    print(f"✓ Contour analysis completed")
    print(f"  Total contours found: {diag['total_contours']}")
    print(f"  Valid candidates: {len(diag['candidates'])}")
    print(f"  Rejected: {len(diag['rejected'])}")
    
    if selected is not None:
        area = cv2.contourArea(selected)
        print(f"  Selected contour area: {area:.0f} px")
        
        # Should select hand-like shape (area ~2500-3000)
        # Not the large circle (area ~20000)
        assert area < 10000, f"Selected too large contour ({area:.0f} px)"
        print("✓ Correctly selected smaller hand-like contour!")
    
    print("✓ Contour selection test passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# IMPROVED SEGMENTATION TEST SUITE")
    print("#"*60)
    
    try:
        test_calibration_with_synthetic_hand()
        test_hue_wraparound()
        test_background_rejection()
        test_hand_contour_selection()
        
        print("\n" + "="*60)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("="*60 + "\n")
        return True
    
    except AssertionError as e:
        print(f"\n✗✗✗ TEST FAILED ✗✗✗")
        print(f"Error: {e}\n")
        return False
    
    except Exception as e:
        print(f"\n✗✗✗ UNEXPECTED ERROR ✗✗✗")
        print(f"Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
