"""
Gesture history module for temporal smoothing and debouncing.
Prevents rapid gesture label switching by requiring consensus across frames.
"""

from collections import deque
from typing import Optional, List


class GestureHistory:
    """
    Track gesture predictions over time and output smoothed/stable gesture.
    Uses consensus voting: only output gesture when multiple consecutive frames agree.
    """
    
    def __init__(self, buffer_size: int = 10, consensus_threshold: int = 5):
        """
        Initialize gesture history tracker.
        
        Args:
            buffer_size: Number of frames to track (sliding window)
            consensus_threshold: Minimum number of frames that must agree for output
        """
        self.buffer_size = buffer_size
        self.consensus_threshold = min(consensus_threshold, buffer_size)
        self.history = deque(maxlen=buffer_size)
        self.last_output_gesture = None
    
    def add_frame(self, gesture_label: Optional[str]) -> None:
        """
        Add a gesture prediction from the current frame.
        
        Args:
            gesture_label: Predicted gesture ("Fist", "Open Palm", "One Finger", "Two Fingers", None)
        """
        self.history.append(gesture_label)
    
    def get_smoothed_gesture(self) -> Optional[str]:
        """
        Get the smoothed gesture based on consensus voting.
        
        Returns:
            Gesture label if consensus reached, None otherwise.
            Maintains last known gesture while waiting for consensus on new gesture.
        """
        if not self.history:
            return None
        
        # Count occurrences of each gesture in history
        gesture_counts = {}
        for gesture in self.history:
            if gesture is not None:
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        # Find if any gesture has reached consensus threshold
        for gesture, count in gesture_counts.items():
            if count >= self.consensus_threshold:
                self.last_output_gesture = gesture
                return gesture
        
        # No consensus yet - if we have a "No Hand" (None) consensus, return None
        # Otherwise, keep last known gesture
        none_count = sum(1 for g in self.history if g is None)
        if none_count >= self.consensus_threshold:
            self.last_output_gesture = None
            return None
        
        # No consensus - return None (don't output anything yet)
        return None
    
    def get_history(self) -> List[Optional[str]]:
        """Get current gesture history."""
        return list(self.history)
    
    def get_confidence(self, gesture: str) -> float:
        """
        Get confidence (0.0 to 1.0) for a specific gesture in current history.
        
        Args:
            gesture: Gesture label to check confidence for
        
        Returns:
            Ratio of frames showing this gesture to total frames
        """
        if not self.history:
            return 0.0
        
        count = sum(1 for g in self.history if g == gesture)
        return count / len(self.history)
    
    def reset(self) -> None:
        """Clear history (e.g., when hand is lost)."""
        self.history.clear()
        self.last_output_gesture = None
    
    def set_parameters(self, buffer_size: Optional[int] = None, 
                       consensus_threshold: Optional[int] = None) -> None:
        """Update smoothing parameters."""
        if buffer_size is not None:
            self.buffer_size = buffer_size
            # Recreate deque with new size
            old_history = list(self.history)
            self.history = deque(old_history, maxlen=buffer_size)
        
        if consensus_threshold is not None:
            self.consensus_threshold = min(consensus_threshold, self.buffer_size)
