"""
Hand detection module for isolating hand contours from skin masks.
Finds, filters, scores, and extracts hand regions from binary masks.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.results import HandDetectionResult


class HandDetector:
    """Detect and isolate hand contour from skin segmentation mask."""

    def __init__(
        self,
        min_contour_area: float = 500.0,
        min_contour_area_ratio: float = 0.005,
        max_contour_area_ratio: float = 0.80,
        min_contour_score: float = 0.5,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> None:
        """
        Initialize hand detector.

        Args:
            min_contour_area: Absolute minimum contour area to consider as a hand
            min_contour_area_ratio: Resolution-scaled minimum area fraction
            max_contour_area_ratio: Reject near-full-frame skin regions as background
            min_contour_score: Minimum required score to accept a contour as a hand
            frame_width: Expected frame width for scaling/validation
            frame_height: Expected frame height for scaling/validation
        """
        self.min_contour_area = min_contour_area
        self.min_contour_area_ratio = max(0.0, min_contour_area_ratio)
        self.max_contour_area_ratio = min(1.0, max(0.01, max_contour_area_ratio))
        self.min_contour_score = min_contour_score
        self.frame_width = frame_width
        self.frame_height = frame_height

    def set_min_contour_score(self, score: float) -> None:
        """Update minimum contour score threshold."""
        self.min_contour_score = score

    def _validate_mask_input(self, mask: np.ndarray) -> None:
        """Validate public input mask before calling OpenCV operations."""
        if mask is None:
            raise TypeError("mask cannot be None")
        if not isinstance(mask, np.ndarray):
            raise TypeError(f"mask must be a numpy.ndarray, got {type(mask).__name__}")
        if mask.size == 0:
            raise ValueError("mask cannot be empty")
        if mask.ndim != 2:
            raise ValueError(f"mask must be a 2D single-channel image, got {mask.ndim}D array")
        if mask.shape[0] == 0 or mask.shape[1] == 0:
            raise ValueError(f"mask dimensions must be non-zero, got shape {mask.shape}")

    def find_hand_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the most plausible hand contour in mask.

        Args:
            mask: Binary mask from skin detection

        Returns:
            Most plausible hand contour or None if no candidate passes minimum score
        """
        res = self.process_mask(mask)
        return res.selected_contour

    def analyze_mask(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Analyze binary mask and return candidate contours, scoring, and diagnostics.
        """
        self._validate_mask_input(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_h, frame_w = mask.shape[:2]
        frame_area = float(frame_h * frame_w)
        minimum_area = max(self.min_contour_area, frame_area * self.min_contour_area_ratio)
        maximum_area = frame_area * self.max_contour_area_ratio

        candidates = []
        rejected = []
        plausible_contours = []

        for idx, contour in enumerate(contours):
            area = float(cv2.contourArea(contour))
            contour_info = {
                "index": idx,
                "area": area,
                "contour": contour,
            }

            # Area filter
            if not (minimum_area <= area <= maximum_area):
                rejected.append({**contour_info, "reason": "area_filter"})
                continue

            # Bounding box & aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                rejected.append({**contour_info, "reason": "invalid_bounding_box"})
                continue

            aspect_ratio = float(w) / float(h)

            # Near-full-frame background border touch / strip filter
            if (
                w >= frame_w * 0.95
                or h >= frame_h * 0.95
                or aspect_ratio < 0.10
                or aspect_ratio > 8.5
            ):
                rejected.append({**contour_info, "reason": "extreme_aspect_ratio_or_strip"})
                continue

            # Implausible hand aspect ratio check (hands are rarely narrower than 0.20 or wider than 4.50)
            if aspect_ratio < 0.20 or aspect_ratio > 4.50:
                rejected.append({**contour_info, "reason": "implausible_aspect_ratio"})
                continue

            perimeter = float(cv2.arcLength(contour, closed=True))
            if perimeter <= 0:
                rejected.append({**contour_info, "reason": "zero_perimeter"})
                continue

            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            rect_area = float(w * h)
            extent = area / rect_area

            hull = cv2.convexHull(contour, returnPoints=True)
            hull_area = float(cv2.contourArea(hull)) if len(hull) >= 3 else area
            solidity = area / hull_area if hull_area > 0 else 1.0

            # Count convexity defects for finger detection
            defect_count = 0
            try:
                hull_indices = cv2.convexHull(contour, returnPoints=False)
                if hull_indices is not None and len(hull_indices) >= 3:
                    defects = cv2.convexityDefects(contour, hull_indices)
                    if defects is not None:
                        for d in defects.reshape(-1, 4):
                            depth = d[3] / 256.0
                            if depth >= 10.0:
                                defect_count += 1
            except cv2.error:
                defect_count = 0

            # Contour scoring
            score = 0.0
            rejection_reasons = []

            # 1. Circularity check: smooth circles are usually background blobs
            if circularity > 0.92 and solidity > 0.92:
                score -= 4.0
                rejection_reasons.append("too_circular_blob")
            elif 0.15 <= circularity <= 0.85:
                score += 2.0
            else:
                score += 0.5

            # 2. Aspect ratio
            if 0.3 <= aspect_ratio <= 3.5:
                score += 1.5
            else:
                score -= 1.5
                rejection_reasons.append("bad_aspect_ratio")

            # 3. Solidity
            if 0.4 <= solidity <= 0.95:
                score += 1.5
            elif solidity < 0.3:
                score -= 2.0
                rejection_reasons.append("low_solidity")

            # 4. Extent
            if 0.25 <= extent <= 0.85:
                score += 1.0
            else:
                score -= 1.0
                rejection_reasons.append("bad_extent")

            # 5. Defects bonus
            if defect_count >= 1:
                score += 2.0

            # 6. Area ratio score
            area_ratio = area / frame_area
            if 0.01 <= area_ratio <= 0.45:
                score += 1.5
            elif area_ratio > 0.65:
                score -= 3.0
                rejection_reasons.append("too_large")

            cand_entry = {
                "contour": contour,
                "area": area,
                "perimeter": perimeter,
                "score": score,
                "circularity": circularity,
                "aspect_ratio": aspect_ratio,
                "solidity": solidity,
                "extent": extent,
                "defect_count": defect_count,
                "rejection_reasons": rejection_reasons,
                "bbox": (int(x), int(y), int(w), int(h)),
            }

            candidates.append(cand_entry)
            plausible_contours.append(contour)

        selected_contour = None
        selected_cand = None

        if candidates:
            # Sort candidates by score (highest first)
            candidates.sort(key=lambda c: c["score"], reverse=True)
            best_candidate = candidates[0]
            # Check against configurable minimum contour score threshold
            if best_candidate["score"] >= self.min_contour_score:
                selected_cand = best_candidate
                selected_contour = selected_cand["contour"]
            else:
                rejected.append({**best_candidate, "reason": "below_min_contour_score"})

        return {
            "contour_count": len(contours),
            "plausible_contour_count": len(plausible_contours),
            "candidates": candidates,
            "rejected": rejected,
            "minimum_area": minimum_area,
            "maximum_area": maximum_area,
            "selected_contour": selected_contour,
            "selected_candidate": selected_cand,
        }

    def process_mask(self, mask: np.ndarray) -> HandDetectionResult:
        """Process mask and return detailed HandDetectionResult."""
        analysis = self.analyze_mask(mask)
        selected = analysis.get("selected_contour")
        cand = analysis.get("selected_candidate")

        if selected is None or cand is None:
            return HandDetectionResult(
                selected_contour=None,
                score=0.0,
                area=0.0,
                perimeter=0.0,
                bounding_box=None,
                candidates=analysis.get("candidates", []),
                rejected=analysis.get("rejected", []),
                is_hand_detected=False,
            )

        return HandDetectionResult(
            selected_contour=selected,
            score=cand.get("score", 0.0),
            area=cand.get("area", 0.0),
            perimeter=cand.get("perimeter", 0.0),
            bounding_box=cand.get("bbox"),
            candidates=analysis.get("candidates", []),
            rejected=analysis.get("rejected", []),
            is_hand_detected=True,
        )

    def get_diagnostic_info(self, mask: np.ndarray) -> Dict[str, Any]:
        """Get full diagnostic dictionary for contour analysis."""
        analysis = self.analyze_mask(mask)
        selected = analysis.get("selected_contour")
        selected_cand = analysis.get("selected_candidate")

        selected_area = float(cv2.contourArea(selected)) if selected is not None else 0.0
        selected_perimeter = float(cv2.arcLength(selected, True)) if selected is not None else 0.0
        selected_score = selected_cand.get("score", 0.0) if selected_cand is not None else 0.0

        return {
            "total_contours": analysis["contour_count"],
            "candidate_count": len(analysis["candidates"]),
            "candidates": analysis["candidates"],
            "rejected": analysis["rejected"],
            "selected": selected,
            "selected_area": selected_area,
            "selected_perimeter": selected_perimeter,
            "selected_score": selected_score,
        }

    def filter_contours(
        self, contours_or_mask: Union[List[np.ndarray], np.ndarray], min_area: Optional[float] = None
    ) -> List[np.ndarray]:
        """Filter list of contours or mask contours by minimum area."""
        if min_area is None:
            min_area = self.min_contour_area

        if isinstance(contours_or_mask, np.ndarray) and contours_or_mask.ndim == 2:
            contours, _ = cv2.findContours(contours_or_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elif isinstance(contours_or_mask, list):
            contours = contours_or_mask
        else:
            return []

        return [c for c in contours if cv2.contourArea(c) >= min_area]

    def get_bounding_box(self, contour: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box (x, y, width, height) of contour."""
        x, y, w, h = cv2.boundingRect(contour)
        return int(x), int(y), int(w), int(h)

    def get_contour_area(self, contour: np.ndarray) -> float:
        """Get contour area."""
        return float(cv2.contourArea(contour))

    def get_contour_perimeter(self, contour: np.ndarray) -> float:
        """Get contour perimeter."""
        return float(cv2.arcLength(contour, closed=True))

    def set_min_contour_area(self, min_area: float) -> None:
        """Update minimum contour area threshold."""
        self.min_contour_area = max(1.0, min_area)

    def draw_contour(
        self,
        frame: np.ndarray,
        contour: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw contour on frame."""
        output = frame.copy()
        cv2.drawContours(output, [contour], 0, color, thickness)
        return output

    def draw_bounding_box(
        self,
        frame: np.ndarray,
        contour: np.ndarray,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding box on frame."""
        output = frame.copy()
        x, y, w, h = self.get_bounding_box(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
        return output

    def draw_both(
        self,
        frame: np.ndarray,
        contour: np.ndarray,
        contour_color: Tuple[int, int, int] = (0, 255, 0),
        box_color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw both contour and bounding box on frame."""
        output = self.draw_contour(frame, contour, contour_color, thickness)
        output = self.draw_bounding_box(output, contour, box_color, thickness)
        return output
