"""
Test Suite for Context Management.

Tests context accumulation and compaction functionality.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.eval_framework.utils.context import ContextManager


class TestContextManager:
    """Test context manager functionality."""

    def test_add_interaction(self):
        """Test adding interactions to history."""
        manager = ContextManager()

        manager.add_interaction(
            context="Test context",
            response="Test response",
            evaluation_summary="Score: 0.8"
        )

        assert len(manager.history) == 1
        assert manager.history[0]["context"] == "Test context"
        assert manager.history[0]["response"] == "Test response"

    def test_response_truncation(self):
        """Test that long responses are truncated."""
        manager = ContextManager()

        long_response = "x" * 1000

        manager.add_interaction(
            context="Test",
            response=long_response
        )

        # Should be truncated to 500 chars + "..."
        assert len(manager.history[0]["response"]) == 503

    def test_get_accumulated_context_empty(self):
        """Test accumulated context when empty."""
        manager = ContextManager()

        context = manager.get_accumulated_context()

        assert context == ""

    def test_get_accumulated_context_with_history(self):
        """Test accumulated context with history."""
        manager = ContextManager()

        manager.add_interaction("Context 1", "Response 1")
        manager.add_interaction("Context 2", "Response 2")

        context = manager.get_accumulated_context()

        assert "Context 1" in context or "Context 2" in context
        assert "Response 1" in context or "Response 2" in context

    def test_needs_compaction_by_items(self):
        """Test compaction trigger by item count."""
        manager = ContextManager(max_history_items=5)

        # Add items below threshold
        for i in range(5):
            manager.history.append({"context": f"c{i}", "response": f"r{i}"})

        assert not manager._needs_compaction()

        # Add one more to exceed threshold
        manager.history.append({"context": "c5", "response": "r5"})

        assert manager._needs_compaction()

    def test_needs_compaction_by_chars(self):
        """Test compaction trigger by character count."""
        manager = ContextManager(max_context_chars=100)

        # Add item with content under threshold
        manager.history.append({"context": "short", "response": "text"})

        assert not manager._needs_compaction()

        # Add item to exceed threshold
        manager.history.append({
            "context": "x" * 50,
            "response": "y" * 50
        })

        assert manager._needs_compaction()

    def test_clear(self):
        """Test clearing context."""
        manager = ContextManager()

        manager.add_interaction("Test", "Response")
        manager.compacted_history = "Summary"
        manager.context_version = 5

        manager.clear()

        assert len(manager.history) == 0
        assert manager.compacted_history == ""
        assert manager.context_version == 0

    def test_get_stats(self):
        """Test getting context stats."""
        manager = ContextManager()

        manager.add_interaction("Test", "Response")
        manager.compacted_history = "Summary"
        manager.context_version = 2

        stats = manager.get_stats()

        assert stats["history_items"] == 1
        assert stats["compacted_history_chars"] == 7
        assert stats["context_version"] == 2


class TestContextCompaction:
    """Test context compaction functionality."""

    @patch('src.eval_framework.utils.context.anthropic.Anthropic')
    def test_compaction_creates_summary(self, mock_anthropic):
        """Test that compaction creates a summary."""
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Summarized history")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        manager = ContextManager(
            max_history_items=3,
            keep_recent_items=1
        )
        manager.client = mock_client

        # Add items to trigger compaction
        for i in range(5):
            manager.history.append({
                "context": f"context {i}",
                "response": f"response {i}",
                "evaluation_summary": "test"
            })

        # Trigger compaction
        manager._compact_history()

        # Should have called the API
        assert mock_client.messages.create.called

    def test_compaction_preserves_recent_items(self):
        """Test that recent items are preserved during compaction."""
        manager = ContextManager(keep_recent_items=2)

        # Manually set history
        manager.history = [
            {"context": "old1", "response": "r1"},
            {"context": "old2", "response": "r2"},
            {"context": "recent1", "response": "r3"},
            {"context": "recent2", "response": "r4"},
        ]

        # Get items to compact
        keep_count = manager.keep_recent_items
        items_to_compact = manager.history[:-keep_count]

        assert len(items_to_compact) == 2
        assert items_to_compact[0]["context"] == "old1"
        assert items_to_compact[1]["context"] == "old2"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
