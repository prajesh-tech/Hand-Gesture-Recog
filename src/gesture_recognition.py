"""
Gesture recognition module using classical computer vision features.
Extracts geometric features from hand contours and classifies into 4 gestures.
Features are measurable and tunable, not hard-coded assumptions.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any


class GestureRecognizer:
    """
    Classify hand gestures using geometric features.
    
    Gesture definitions (based on measurable features):
    - Fist: Compact, high solidity, few convex defects
    - Open Palm: Low solidity, many defects (fingers), large area
    - One Finger: Few defects, elongated, small area
    - Two Fingers: Moderate defects, moderate area
    """
    
    # Tunable thresholds - these are starting points, expect adjustment during testing
    THRESHOLDS = {
        'min_hand_area': 300,          # Minimum contour area to be considered a hand
        'aspect_ratio_min': 0.3,       # Min width/height ratio (elongated)
        'aspect_ratio_max': 3.5,       # Max width/height ratio (not too stretched)
        
        # Fist recognition thresholds
        'fist_solidity_min': 0.75,     # High solidity (compact shape)
        'fist_defects_max': 3,         # Few convex defects
        'fist_extent_min': 0.5,        # Fills bounding box well
        
        # Open Palm thresholds
        'palm_solidity_max': 0.65,     # Low solidity (fingers create indentations)
        'palm_defects_min': 6,         # Many defects from fingers
        'palm_extent_min': 0.3,        # Lower extent (irregular shape)
        
        # One Finger thresholds
        'one_finger_area_max': 15000,  # Smaller area (single finger)
        'one_finger_defects_max': 3,   # Few defects
        'one_finger_defects_min': 0,
        
        # Two Fingers thresholds
        'two_fingers_area_max': 25000, # Moderate area
        'two_fingers_defects_min': 2,  # At least 2 defects
        'two_fingers_defects_max': 4,  # Not too many
    }
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize gesture recognizer.
        
        Args:
            thresholds: Optional dict to override default thresholds
        """
        self.thresholds = self.THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)
    
    def extract_features(self, contour: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Extract geometric features from hand contour.
        
        Args:
            contour: Hand contour from cv2.findContours
        
        Returns:
            Dict with features: area, perimeter, solidity, extent, aspect_ratio, 
            convexity_defects_count, convex_hull, or None if contour invalid
        """
        try:
            # Basic properties
            area = cv2.contourArea(contour)
            
            # Filter by minimum area
            if area < self.thresholds['min_hand_area']:
                return None
            
            perimeter = cv2.arcLength(contour, closed=True)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            if w == 0 or h == 0:
                return None
            
            aspect_ratio = float(w) / float(h)
            bbox_area = w * h
            
            # Extent (ratio of contour area to bounding box area)
            extent = area / bbox_area if bbox_area > 0 else 0
            
            # Convex hull and solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull) if len(hull) > 0 else area
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Convexity defects (indentations between fingers)
            defects_count = 0
            try:
                if len(hull) > 3 and len(contour) > 3:
                    defects = cv2.convexityDefects(contour, hull)
                    # Count significant defects (depth > threshold)
                    if defects is not None:
                        for defect in defects:
                            s, e, f, d = defect[0]
                            # d is distance (scaled by 256), threshold at ~5000 to ignore noise
                            if d > 5000:
                                defects_count += 1
            except cv2.error:
                # Handle edge case where convexityDefects fails
                defects_count = 0
            
            # Circularity (compactness) - ratio of perimeter to area
            circularity = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
            
            return {
                'area': float(area),
                'perimeter': float(perimeter),
                'solidity': float(solidity),
                'extent': float(extent),
                'aspect_ratio': float(aspect_ratio),
                'convexity_defects_count': int(defects_count),
                'circularity': float(circularity),
                'hull': hull,
                'hull_area': float(hull_area),
                'bbox': (x, y, w, h),
            }
        
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None
    
    def recognize_gesture(self, contour: np.ndarray) -> Optional[str]:
        """
        Classify gesture from hand contour.
        
        Args:
            contour: Hand contour
        
        Returns:
            Gesture label: "Fist", "Open Palm", "One Finger", "Two Fingers", or None
        """
        features = self.extract_features(contour)
        if features is None:
            return None
        
        return self._classify_by_features(features)
    
    def _classify_by_features(self, features: Dict[str, Any]) -> Optional[str]:
        """
        Classify gesture based on extracted features.
        
        This is where tuning happens: adjust thresholds based on testing.
        
        Args:
            features: Feature dict from extract_features
        
        Returns:
            Gesture label or None
        """
        area = features['area']
        solidity = features['solidity']
        extent = features['extent']
        aspect_ratio = features['aspect_ratio']
        defects = features['convexity_defects_count']
        
        # Validate aspect ratio (avoid extreme shapes)
        if aspect_ratio < self.thresholds['aspect_ratio_min'] or aspect_ratio > self.thresholds['aspect_ratio_max']:
            return None
        
        # Classification logic (order matters - check most distinctive first)
        
        # FIST: High solidity, few defects, compact
        if (solidity >= self.thresholds['fist_solidity_min'] and
            defects <= self.thresholds['fist_defects_max'] and
            extent >= self.thresholds['fist_extent_min']):
            return "Fist"
        
        # OPEN PALM: Low solidity, many defects, larger area
        if (solidity <= self.thresholds['palm_solidity_max'] and
            defects >= self.thresholds['palm_defects_min']):
            return "Open Palm"
        
        # ONE FINGER: Small area, very few defects, elongated
        if (area <= self.thresholds['one_finger_area_max'] and
            defects >= self.thresholds['one_finger_defects_min'] and
            defects <= self.thresholds['one_finger_defects_max'] and
            aspect_ratio > 0.8):  # More elongated
            return "One Finger"
        
        # TWO FINGERS: Moderate area, 2-4 defects
        if (area <= self.thresholds['two_fingers_area_max'] and
            defects >= self.thresholds['two_fingers_defects_min'] and
            defects <= self.thresholds['two_fingers_defects_max']):
            return "Two Fingers"
        
        # Default: unknown gesture
        return None
    
    def set_threshold(self, key: str, value: float) -> None:
        """Update a single threshold value."""
        if key in self.thresholds:
            self.thresholds[key] = value
    
    def update_thresholds(self, thresholds_dict: Dict[str, float]) -> None:
        """Update multiple threshold values."""
        self.thresholds.update(thresholds_dict)
    
    def get_threshold(self, key: str) -> Optional[float]:
        """Get current threshold value."""
        return self.thresholds.get(key)
    
    def get_all_thresholds(self) -> Dict[str, float]:
        """Get all current thresholds."""
        return self.thresholds.copy()
