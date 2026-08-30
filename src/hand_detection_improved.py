"""
Improved hand detection with intelligent contour selection.
Considers multiple criteria instead of blindly selecting the largest contour.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any


class HandDetectorImproved:
    """
    Enhanced hand detector with:
    - Smart contour selection (not just largest)
    - Rejection of implausible shapes
    - Diagnostic reporting
    """
    
    def __init__(self, min_contour_area: float = 500, max_contour_area: Optional[float] = None,
                 frame_width: int = 640, frame_height: int = 480):
        """
        Initialize hand detector.
        
        Args:
            min_contour_area: Minimum contour area to consider
            max_contour_area: Maximum contour area to consider (reject near-fullframe)
            frame_width: Frame width (for validation)
            frame_height: Frame height (for validation)
        """
        self.min_contour_area = min_contour_area
        
        # Max area = ~70% of frame (conservative)
        if max_contour_area is None:
            max_contour_area = 0.7 * frame_width * frame_height
        self.max_contour_area = max_contour_area
        
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    def find_hand_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the most plausible hand contour using smart selection.
        
        Args:
            mask: Binary mask from skin detection
        
        Returns:
            Best contour array or None
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter and score contours
        candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Basic area filtering
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
            
            # Compute shape metrics
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 10:
                continue
            
            # Circularity (0-1, 1 = perfect circle)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # Aspect ratio
            rect = cv2.boundingRect(contour)
            x, y, w, h = rect
            aspect_ratio = w / h if h > 0 else 0
            
            # Solidity (compactness)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 1
            solidity = area / hull_area
            
            # Extent (ratio of contour area to bounding rect)
            rect_area = w * h if w > 0 and h > 0 else 1
            extent = area / rect_area
            
            # Score the contour
            score = 0
            rejection_reasons = []
            
            # ✨ IMPROVED SCORING: Prefer hand-like shapes
            
            # 1. Too circular? (circles are less hand-like)
            #    Good hands have circularity 0.3-0.7
            if circularity < 0.3:
                score += 2.0  # Very non-circular = good for hands
            elif circularity < 0.7:
                score += 1.5  # Moderate circularity
            elif circularity < 0.9:
                score += 0.0  # Getting circular
            else:
                score -= 2.0  # Very circular (background blob)
                rejection_reasons.append("too_circular")
            
            # 2. Aspect ratio (hands are usually wider than tall)
            if 0.3 <= aspect_ratio <= 3.5:
                score += 1.5
            else:
                score -= 1.5
                rejection_reasons.append("bad_aspect_ratio")
            
            # 3. Solidity (hands have moderate solidity)
            if solidity > 0.4:
                score += 1.0 * min(solidity / 0.8, 1.0)  # Cap at solidity=0.8
            else:
                score -= 1.0
                rejection_reasons.append("low_solidity")
            
            # 4. Extent (hands should fill their bounding box moderately)
            if extent > 0.3:
                score += 1.0 * min(extent, 1.0)
            else:
                score -= 1.0
                rejection_reasons.append("low_extent")
            
            # 5. Area (prefer medium-sized contours, reject very large)
            area_ratio = area / self.max_contour_area
            if 0.05 < area_ratio < 0.5:  # Prefermoderate sizes
                score += 1.5
            elif area_ratio < 0.05:
                score += 0.5  # Small but valid
            elif area_ratio < 0.75:
                score -= 0.5  # Getting large
            else:
                score -= 3.0  # Very large, likely false positive
                rejection_reasons.append("too_large")
            
            candidates.append({
                'contour': contour,
                'area': area,
                'score': score,
                'circularity': circularity,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'extent': extent,
                'rejection_reasons': rejection_reasons
            })
        
        if not candidates:
            return None
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Return the best candidate
        best = candidates[0]
        return best['contour']
    
    def get_diagnostic_info(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Get diagnostic info about contours found in mask.
        
        Args:
            mask: Binary mask
        
        Returns:
            Diagnostic dictionary
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        info = {
            'total_contours': len(contours),
            'candidates': [],
            'rejected': [],
            'selected': None
        }
        
        if not contours:
            return info
        
        # Analyze all contours
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            contour_info = {
                'index': idx,
                'area': float(area),
                'perimeter': float(perimeter)
            }
            
            # Filter
            if area < self.min_contour_area or area > self.max_contour_area:
                info['rejected'].append({**contour_info, 'reason': 'area_filter'})
                continue
            
            info['candidates'].append(contour_info)
        
        return info
    
    def filter_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Get all valid contours (before scoring).
        
        Args:
            mask: Binary mask
        
        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area <= area <= self.max_contour_area:
                filtered.append(contour)
        
        return filtered
    
    def get_bounding_box(self, contour: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box (x, y, w, h) for contour."""
        return cv2.boundingRect(contour)
    
    def get_contour_area(self, contour: np.ndarray) -> float:
        """Get area of contour."""
        return cv2.contourArea(contour)
    
    def get_contour_perimeter(self, contour: np.ndarray) -> float:
        """Get perimeter of contour."""
        return cv2.arcLength(contour, True)
    
    def draw_contour(self, frame: np.ndarray, contour: np.ndarray,
                    color: Tuple[int, int, int] = (0, 255, 0),
                    thickness: int = 2) -> np.ndarray:
        """Draw contour on frame."""
        return cv2.drawContours(frame, [contour], 0, color, thickness)
    
    def draw_bounding_box(self, frame: np.ndarray, contour: np.ndarray,
                         color: Tuple[int, int, int] = (255, 0, 0),
                         thickness: int = 2) -> np.ndarray:
        """Draw bounding box on frame."""
        x, y, w, h = self.get_bounding_box(contour)
        return cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    def draw_both(self, frame: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """Draw both contour and bounding box."""
        frame = self.draw_bounding_box(frame, contour, (255, 0, 0), 2)
        frame = self.draw_contour(frame, contour, (0, 255, 0), 2)
        return frame
    
    def set_min_contour_area(self, area: float) -> None:
        """Set minimum contour area."""
        self.min_contour_area = area
