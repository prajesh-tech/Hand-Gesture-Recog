"""
Hand detection module for isolating hand contours from skin masks.
Finds, filters, and extracts hand region from binary masks.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple


class HandDetector:
    """Detect and isolate hand contour from skin segmentation mask."""
    
    def __init__(self, min_contour_area: float = 500):
        """
        Initialize hand detector.
        
        Args:
            min_contour_area: Minimum contour area to consider as a valid hand
        """
        self.min_contour_area = min_contour_area
    
    def find_hand_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the largest contour in mask (assumed to be the hand).
        
        Args:
            mask: Binary mask from skin detection
        
        Returns:
            Largest contour (numpy array) or None if no valid contour found
        """
        # Find all contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours by minimum area
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]
        
        if not valid_contours:
            return None
        
        # Return the largest contour (most likely the hand)
        hand_contour = max(valid_contours, key=cv2.contourArea)
        
        return hand_contour
    
    def filter_contours(self, contours: List[np.ndarray], min_area: Optional[float] = None) -> List[np.ndarray]:
        """
        Filter contours by minimum area.
        
        Args:
            contours: List of contours
            min_area: Minimum area threshold (uses self.min_contour_area if None)
        
        Returns:
            Filtered list of contours
        """
        if min_area is None:
            min_area = self.min_contour_area
        
        return [c for c in contours if cv2.contourArea(c) >= min_area]
    
    def get_bounding_box(self, contour: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get bounding box of contour.
        
        Args:
            contour: Input contour
        
        Returns:
            Tuple of (x, y, width, height)
        """
        x, y, w, h = cv2.boundingRect(contour)
        return int(x), int(y), int(w), int(h)
    
    def get_contour_area(self, contour: np.ndarray) -> float:
        """Get area of contour."""
        return float(cv2.contourArea(contour))
    
    def get_contour_perimeter(self, contour: np.ndarray) -> float:
        """Get perimeter of contour."""
        return float(cv2.arcLength(contour, closed=True))
    
    def set_min_contour_area(self, min_area: float) -> None:
        """Update minimum contour area threshold."""
        self.min_contour_area = max(1.0, min_area)
    
    def draw_contour(self, frame: np.ndarray, contour: np.ndarray, 
                     color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw contour on frame.
        
        Args:
            frame: Input frame (BGR)
            contour: Contour to draw
            color: BGR color
            thickness: Line thickness
        
        Returns:
            Frame with drawn contour
        """
        frame = frame.copy()
        cv2.drawContours(frame, [contour], 0, color, thickness)
        return frame
    
    def draw_bounding_box(self, frame: np.ndarray, contour: np.ndarray, 
                         color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw bounding box on frame.
        
        Args:
            frame: Input frame (BGR)
            contour: Contour for which to draw bounding box
            color: BGR color
            thickness: Line thickness
        
        Returns:
            Frame with drawn bounding box
        """
        frame = frame.copy()
        x, y, w, h = self.get_bounding_box(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        return frame
    
    def draw_both(self, frame: np.ndarray, contour: np.ndarray,
                  contour_color: Tuple[int, int, int] = (0, 255, 0),
                  box_color: Tuple[int, int, int] = (255, 0, 0),
                  thickness: int = 2) -> np.ndarray:
        """Draw both contour and bounding box."""
        frame = self.draw_contour(frame, contour, contour_color, thickness)
        frame = self.draw_bounding_box(frame, contour, box_color, thickness)
        return frame
