"""
Unit tests for gesture history (temporal smoothing) module.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gesture_history import GestureHistory


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
        # Add more frames than buffer size
        for i in range(15):
            history.add_frame("Fist")
        
        # Should only keep last 10
        assert len(history.get_history()) <= 10
    
    def test_get_smoothed_gesture_no_consensus(self, history):
        """Test that no gesture is returned without consensus."""
        # Add alternating gestures (no consensus)
        history.add_frame("Fist")
        history.add_frame("Open Palm")
        history.add_frame("Fist")
        
        smoothed = history.get_smoothed_gesture()
        # No gesture reaches 5-frame consensus
        assert smoothed is None
    
    def test_get_smoothed_gesture_with_consensus(self, history):
        """Test that gesture is returned when consensus is reached."""
        # Add 5 identical gestures
        for _ in range(5):
            history.add_frame("Fist")
        
        smoothed = history.get_smoothed_gesture()
        # Should reach consensus
        assert smoothed == "Fist"
    
    def test_consensus_threshold_enforcement(self):
        """Test that consensus threshold is respected."""
        history = GestureHistory(buffer_size=10, consensus_threshold=3)
        
        # Add only 2 identical gestures
        history.add_frame("Open Palm")
        history.add_frame("Open Palm")
        
        smoothed = history.get_smoothed_gesture()
        # Shouldn't reach consensus of 3
        assert smoothed is None
        
        # Add third identical gesture
        history.add_frame("Open Palm")
        smoothed = history.get_smoothed_gesture()
        # Should reach consensus now
        assert smoothed == "Open Palm"
    
    def test_no_hand_detection_consensus(self, history):
        """Test consensus with None (no hand detected)."""
        # Add 5 None values
        for _ in range(5):
            history.add_frame(None)
        
        smoothed = history.get_smoothed_gesture()
        # Should recognize None consensus
        assert smoothed is None
    
    def test_gesture_switch_requires_new_consensus(self, history):
        """Test that switching gestures requires new consensus."""
        # Add 5 "Fist" gestures
        for _ in range(5):
            history.add_frame("Fist")
        
        smoothed = history.get_smoothed_gesture()
        assert smoothed == "Fist"
        
        # Add 1 "Open Palm"
        history.add_frame("Open Palm")
        smoothed = history.get_smoothed_gesture()
        # Should still be Fist (Open Palm not yet consensus)
        assert smoothed == "Fist"
        
        # Add 5 more "Open Palm" (total 6, which outnumbers Fist's 5)
        for _ in range(5):
            history.add_frame("Open Palm")
        
        smoothed = history.get_smoothed_gesture()
        # Now Open Palm should have consensus (6 > 5)
        assert smoothed == "Open Palm"
    
    def test_get_confidence(self, history):
        """Test confidence calculation."""
        # Add 5 Fist out of 10
        for _ in range(5):
            history.add_frame("Fist")
        for _ in range(5):
            history.add_frame("Open Palm")
        
        confidence = history.get_confidence("Fist")
        assert confidence == 0.5
        
        confidence_palm = history.get_confidence("Open Palm")
        assert confidence_palm == 0.5
    
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
        
        # Should be equal but different objects
        assert h1 == h2
        assert h1 is not h2
