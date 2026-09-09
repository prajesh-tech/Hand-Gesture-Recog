"""
Gesture history module for temporal smoothing and debouncing.
Prevents rapid gesture label switching by requiring consensus across frames.
"""

from collections import deque
from typing import ClassVar, Dict, List, Optional

from src.results import GestureHistoryResult


class GestureHistory:
    """
    Track gesture predictions over time and output smoothed/stable gesture.
    Uses sliding-window consensus voting: only output a gesture when enough
    frames in the current window agree.
    """

    GESTURE_ACTIONS: ClassVar[Dict[str, str]] = {
        "Fist": "STOP",
        "Open Palm": "START",
        "One Finger": "SELECT",
        "Two Fingers": "NEXT",
    }

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
        self.last_consensus_gesture = None

    @property
    def last_output_gesture(self) -> Optional[str]:
        """Backward-compatible name for the most recent consensus gesture."""
        return self.last_consensus_gesture

    @last_output_gesture.setter
    def last_output_gesture(self, gesture: Optional[str]) -> None:
        self.last_consensus_gesture = gesture

    def add_frame(self, gesture_label: Optional[str]) -> None:
        """
        Add a gesture prediction from the current frame.

        Args:
            gesture_label: Predicted gesture ("Fist", "Open Palm", "One Finger", "Two Fingers", None)
        """
        # Unknown and No Hand Detected are ambiguous/empty frames, not competing gesture votes.
        if gesture_label in ("Unknown", "No Hand Detected", None):
            self.history.append(None)
        else:
            self.history.append(gesture_label)

    def get_smoothed_gesture(self) -> Optional[str]:
        """
        Get the smoothed gesture based on consensus voting.

        Returns:
            Gesture label if consensus reached, None otherwise.
            Returns None until the current history reaches consensus. There is no
            sticky output while waiting for a new consensus.
        """
        if not self.history:
            return None

        # Count occurrences of each gesture in history
        gesture_counts = {}
        for gesture in self.history:
            if gesture is not None:
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1

        # Find if any gesture has reached consensus threshold.
        # Sort by count descending so the most-agreed-on gesture wins when
        # multiple gestures exceed the threshold simultaneously.
        for gesture, count in sorted(gesture_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= self.consensus_threshold:
                self.last_consensus_gesture = gesture
                return gesture

        # No consensus yet - if we have a "No Hand" (None) consensus, return None
        none_count = sum(1 for g in self.history if g is None)
        if none_count >= self.consensus_threshold:
            self.last_consensus_gesture = None
            return None

        return None

    def process_history(self, gesture_label: Optional[str]) -> GestureHistoryResult:
        """Add a raw frame result and return the current strict-debounce state."""
        self.add_frame(gesture_label)
        smoothed = self.get_smoothed_gesture()
        action = self.GESTURE_ACTIONS.get(smoothed, "") if smoothed else ""
        temporal_confidence = self.get_confidence(smoothed) if smoothed else 0.0

        return GestureHistoryResult(
            smoothed_gesture=smoothed,
            action=action,
            history=self.get_history(),
            temporal_confidence=temporal_confidence,
        )

    def get_history(self) -> List[Optional[str]]:
        """Get current gesture history."""
        return list(self.history)

    def get_confidence(self, gesture: Optional[str]) -> float:
        """
        Get confidence (0.0 to 1.0) for a specific gesture in current history.

        Args:
            gesture: Gesture label to check confidence for

        Returns:
            Ratio of frames showing this gesture to total frames
        """
        if not self.history or gesture is None:
            return 0.0

        count = sum(1 for g in self.history if g == gesture)
        return count / len(self.history)

    def reset(self) -> None:
        """Clear history (e.g., when hand is lost)."""
        self.history.clear()
        self.last_consensus_gesture = None

    def set_parameters(
        self, buffer_size: Optional[int] = None, consensus_threshold: Optional[int] = None
    ) -> None:
        """Update smoothing parameters."""
        if buffer_size is not None:
            self.buffer_size = buffer_size
            old_history = list(self.history)
            self.history = deque(old_history, maxlen=buffer_size)

        if consensus_threshold is not None:
            self.consensus_threshold = min(consensus_threshold, self.buffer_size)
