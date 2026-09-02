"""
Unit tests for gesture history (temporal smoothing) module and result dataclasses.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gesture_history import GestureHistory
from src.results import GestureHistoryResult


class TestGestureHistory:
    """Test GestureHistory class."""

    @pytest.fixture
    def history(self):
        """Create a history tracker."""
        return GestureHistory(buffer_size=10, consensus_threshold=5)

    def test_initialization(self, history):
        """Test history initialization."""
        assert history.buffer_size == 10
        assert history.consensus_threshold == 5
        assert len(history.get_history()) == 0

    def test_add_frame(self, history):
        """Test adding frames to history."""
        history.add_frame("Fist")
        assert len(history.get_history()) == 1

        history.add_frame("Open Palm")
        assert len(history.get_history()) == 2

    def test_buffer_size_limit(self, history):
        """Test that buffer respects size limit."""
        for i in range(15):
            history.add_frame("Fist")

        assert len(history.get_history()) <= 10

    def test_get_smoothed_gesture_no_consensus(self, history):
        """Test that no gesture is returned without consensus."""
        history.add_frame("Fist")
        history.add_frame("Open Palm")
        history.add_frame("Fist")

        smoothed = history.get_smoothed_gesture()
        assert smoothed is None

    def test_get_smoothed_gesture_with_consensus(self, history):
        """Test that gesture is returned when consensus is reached."""
        for _ in range(5):
            history.add_frame("Fist")

        smoothed = history.get_smoothed_gesture()
        assert smoothed == "Fist"

    def test_consensus_threshold_enforcement(self):
        """Test that consensus threshold is respected."""
        history = GestureHistory(buffer_size=10, consensus_threshold=3)

        history.add_frame("Open Palm")
        history.add_frame("Open Palm")

        smoothed = history.get_smoothed_gesture()
        assert smoothed is None

        history.add_frame("Open Palm")
        smoothed = history.get_smoothed_gesture()
        assert smoothed == "Open Palm"

    def test_no_hand_detection_consensus(self, history):
        """Test consensus with None (no hand detected)."""
        for _ in range(5):
            history.add_frame(None)

        smoothed = history.get_smoothed_gesture()
        assert smoothed is None

    def test_gesture_switch_requires_new_consensus(self, history):
        """Test that switching gestures requires new consensus."""
        for _ in range(5):
            history.add_frame("Fist")

        smoothed = history.get_smoothed_gesture()
        assert smoothed == "Fist"

        history.add_frame("Open Palm")
        smoothed = history.get_smoothed_gesture()
        assert smoothed == "Fist"

        for _ in range(5):
            history.add_frame("Open Palm")

        smoothed = history.get_smoothed_gesture()
        assert smoothed == "Open Palm"

    def test_get_confidence(self, history):
        """Test confidence calculation."""
        for _ in range(5):
            history.add_frame("Fist")
        for _ in range(5):
            history.add_frame("Open Palm")

        confidence = history.get_confidence("Fist")
        assert confidence == 0.5

        confidence_palm = history.get_confidence("Open Palm")
        assert confidence_palm == 0.5

    def test_process_history_returns_result_dataclass(self, history):
        """Test process_history returns GestureHistoryResult dataclass."""
        for _ in range(5):
            res = history.process_history("Fist")

        assert isinstance(res, GestureHistoryResult)
        assert res.smoothed_gesture == "Fist"
        assert res.action == "STOP"
        assert res.temporal_confidence == 1.0

    def test_reset(self, history):
        """Test resetting history."""
        history.add_frame("Fist")
        history.add_frame("Fist")

        assert len(history.get_history()) == 2

        history.reset()

        assert len(history.get_history()) == 0
        assert history.last_output_gesture is None

    def test_set_parameters(self, history):
        """Test updating parameters."""
        history.set_parameters(buffer_size=20, consensus_threshold=10)

        assert history.buffer_size == 20
        assert history.consensus_threshold == 10

    def test_get_history_returns_copy(self, history):
        """Test that get_history returns independent list."""
        history.add_frame("Fist")
        h1 = history.get_history()
        h2 = history.get_history()

        assert h1 == h2
        assert h1 is not h2
