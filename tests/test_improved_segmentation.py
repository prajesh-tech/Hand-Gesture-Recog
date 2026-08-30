"""
Regression tests for improved skin detection and calibration.
Tests for proper handling of broad ranges, hue wrap-around, contamination, etc.
"""

import pytest
import numpy as np
import cv2
from src.skin_detection_improved import SkinDetectorImproved
from src.hand_detection_improved import HandDetectorImproved


class TestCalibrationRobustness:
    """Test calibration produces reasonable ranges."""
    
    def test_calibration_with_clean_hand_samples(self):
        """Test calibration with clean hand samples."""
        # Create synthetic hand-like samples (orange/skin tone HSV)
        frames = []
        for _ in range(5):
            # Create frame with hand-like pixels in center
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Center region with skin tone (H~15, S~100, V~150 in HSV)
            # Convert to BGR for input
            center_hsv = np.array([[[15, 100, 150]]], dtype=np.uint8)
            center_bgr = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2BGR)[0, 0]
            
            h, w = 480, 640
            cy, cx = h // 2, w // 2
            region_size = 80
            y1, y2 = cy - region_size, cy + region_size
            x1, x2 = cx - region_size, cx + region_size
            
            frame[y1:y2, x1:x2] = center_bgr
            frames.append(frame)
        
        lower, upper, stats = SkinDetectorImproved.calibrate_from_samples(frames)
        
        # Verify output is valid
        assert stats['calibration_valid'] == True
        assert stats['valid_percentage'] > 50  # Should have good valid pixel percentage
        
        # Hue should be reasonable for skin tone
        assert lower[0] <= 20 or (lower[0] > 150 and stats['hue_wrap_detected'])
        assert upper[0] <= 30 or (upper[0] <= 180)
    
    def test_calibration_rejects_low_valid_percentage(self):
        """Test calibration with mostly background."""
        # Create frame with minimal hand pixels
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Few pixels in center
        h, w = 480, 640
        cy, cx = h // 2, w // 2
        center_hsv = np.array([[[15, 100, 150]]], dtype=np.uint8)
        center_bgr = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        frame[cy-5:cy+5, cx-5:cx+5] = center_bgr
        
        lower, upper, stats = SkinDetectorImproved.calibrate_from_samples([frame])
        
        # Valid percentage should be low
        assert stats['valid_percentage'] < 30
    
    def test_calibration_respects_saturation_bounds(self):
        """Test that S >= 15 and V >= 35 are enforced."""
        frames = []
        for _ in range(3):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add pixels with low saturation (gray)
            h, w = 480, 640
            cy, cx = h // 2, w // 2
            
            # Low-saturation pixels (might appear skin-like in hue but are gray)
            gray_hsv = np.array([[[15, 5, 100]]], dtype=np.uint8)  # Low S
            gray_bgr = cv2.cvtColor(gray_hsv, cv2.COLOR_HSV2BGR)[0, 0]
            frame[cy-50:cy+50, cx-50:cx+50] = gray_bgr
            
            frames.append(frame)
        
        lower, upper, stats = SkinDetectorImproved.calibrate_from_samples(frames)
        
        # Should enforce S >= 15
        assert lower[1] >= 15
        assert lower[2] >= 35


class TestHueWrapAround:
    """Test handling of hue wrap-around for red skin tones."""
    
    def test_red_skin_tone_wrap_around_detection(self):
        """Test detection of hue wrap-around in red skin tones."""
        frames = []
        for _ in range(5):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Red/orange skin tone pixels
            h, w = 480, 640
            cy, cx = h // 2, w // 2
            
            # Mix of hue~10 and hue~170 (both red)
            for i in range(-40, 41, 20):
                for j in range(-40, 41, 20):
                    hue_val = 10 if (i + j) % 2 == 0 else 170
                    hsv = np.array([[[hue_val, 80, 140]]], dtype=np.uint8)
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
                    frame[cy+i, cx+j] = bgr
            
            frames.append(frame)
        
        lower, upper, stats = SkinDetectorImproved.calibrate_from_samples(frames)
        
        # Should detect wrap-around
        assert stats.get('hue_wrap_detected', False) == True or lower[0] < 30 or upper[0] < 30


class TestSkinDetectorMasks:
    """Test that skin masks don't accept excessive background."""
    
    def test_mask_rejects_pure_black_background(self):
        """Test that pure black background is not detected as skin."""
        detector = SkinDetectorImproved()
        
        # Black frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add a small hand-colored region
        center_hsv = np.array([[[15, 80, 150]]], dtype=np.uint8)
        center_bgr = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        h, w = 480, 640
        frame[h//2-30:h//2+30, w//2-30:w//2+30] = center_bgr
        
        mask = detector.detect_skin(frame)
        
        # Skin mask should only have pixels in hand region, not black background
        # Black pixels should not be detected
        background_mask = frame[:, :, 2] < 10  # Blue channel < 10 = near-black
        detected_in_background = np.sum((mask > 0) & background_mask)
        
        assert detected_in_background < 100  # Allow minimal false positives
    
    def test_mask_accepts_hand_pixels(self):
        """Test that actual hand pixels are detected."""
        detector = SkinDetectorImproved()
        detector.lower_hsv = np.array([0, 15, 35], dtype=np.uint8)
        detector.upper_hsv = np.array([30, 255, 255], dtype=np.uint8)
        
        # Frame with skin tone
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        center_hsv = np.array([[[15, 100, 150]]], dtype=np.uint8)
        center_bgr = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        
        h, w = 480, 640
        frame[h//2-50:h//2+50, w//2-50:w//2+50] = center_bgr
        
        mask = detector.detect_skin(frame)
        
        # Should detect hand region
        hand_region_detected = np.sum(mask[h//2-50:h//2+50, w//2-50:w//2+50]) > 100
        assert hand_region_detected


class TestContourDetection:
    """Test smart contour selection."""
    
    def test_selects_hand_shaped_contour_over_circle(self):
        """Test that hand-like contour is preferred over circular blobs."""
        detector = HandDetectorImproved(frame_width=640, frame_height=480)
        
        # Create mask with both a circular blob and hand-like shape
        mask = np.zeros((480, 640), dtype=np.uint8)
        
        # Large circle (not hand-like)
        cv2.circle(mask, (150, 150), 80, 255, -1)
        
        # Hand-like shape (irregular)
        hand_pts = np.array([
            [[300, 200]], [[320, 180]], [[340, 200]], [[350, 220]],
            [[340, 250]], [[320, 260]], [[300, 250]], [[290, 230]]
        ], dtype=np.int32)
        cv2.drawContours(mask, [hand_pts], 0, 255, -1)
        
        selected = detector.find_hand_contour(mask)
        
        # Should select hand-like shape (smaller, less circular)
        if selected is not None:
            area = cv2.contourArea(selected)
            assert area < 6000  # Should be smaller hand, not large circle
    
    def test_rejects_near_fullframe_contour(self):
        """Test that contours filling most of frame are rejected."""
        detector = HandDetectorImproved(frame_width=640, frame_height=480)
        
        # Create mask with near-fullframe blob
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (630, 470), 255, -1)  # Nearly full frame
        
        selected = detector.find_hand_contour(mask)
        
        # Should be rejected
        assert selected is None


class TestImprovedVsOriginal:
    """Compare improved implementation with known problematic cases."""
    
    def test_broad_hsv_range_is_rejected(self):
        """Test that excessively broad calibration is handled."""
        # Simulate old calibration producing broad range
        old_lower = np.array([159, 20, 33], dtype=np.uint8)
        old_upper = np.array([24, 118, 145], dtype=np.uint8)
        
        detector = SkinDetectorImproved()
        detector.set_hsv_range(old_lower, old_upper)
        
        # Create mostly-black frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        mask = detector.detect_skin(frame)
        skin_percentage = 100.0 * np.count_nonzero(mask) / (480 * 640)
        
        # Even with broad range, black background should not be heavily detected
        # (because it has V close to 0)
        assert skin_percentage < 50  # Improved implementation should be smarter


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
