"""
Context Management with Automatic Compaction.

This module provides session context management with automatic
compaction to prevent unbounded context growth.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import anthropic

from .config.settings import settings
from .types import Interaction


class ContextManager:
    """
    Manages session context with automatic compaction.

    Accumulates context across the session and compacts when needed
    to prevent unbounded growth while preserving important information.

    Features:
    - Automatic context accumulation
    - LLM-based summarization for compaction
    - Configurable thresholds
    - Recent items preservation
    """

    def __init__(
        self,
        max_history_items: int = None,
        max_context_chars: int = None,
        compaction_target_chars: int = None,
        keep_recent_items: int = None
    ):
        """
        Initialize the context manager.

        Args:
            max_history_items: Maximum items before compaction triggers.
            max_context_chars: Maximum characters before compaction.
            compaction_target_chars: Target size after compaction.
            keep_recent_items: Number of recent items to keep uncompacted.
        """
        self.max_history_items = max_history_items or settings.context.max_history_items
        self.max_context_chars = max_context_chars or settings.context.max_context_chars
        self.compaction_target_chars = compaction_target_chars or settings.context.compaction_target_chars
        self.keep_recent_items = keep_recent_items or settings.context.keep_recent_items

        self.client = anthropic.Anthropic()

        # Session state
        self.history: List[Dict[str, Any]] = []
        self.compacted_history: str = ""
        self.context_version: int = 0

    def add_interaction(self, context: str, response: str, evaluation_summary: str = "") -> None:
        """
        Add a new interaction (context + response) to history.

        Args:
            context: The context provided for the interaction.
            response: The AI response generated.
            evaluation_summary: Optional summary of the evaluation result.
        """
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "response": response,
            "evaluation_summary": evaluation_summary,
            "index": len(self.history)
        }
        self.history.append(interaction)
        self._check_compaction()

    def get_accumulated_context(self) -> str:
        """
        Get the full accumulated context (compacted + recent).

        Returns:
            String containing the accumulated context.
        """
        parts = []

        if self.compacted_history:
            parts.append("### Previous Session Summary")
            parts.append(self.compacted_history)
            parts.append("")

        if self.history:
            parts.append("### Recent History")
            for item in self.history:
                parts.append(f"TS: {item['timestamp']}")
                if item['context']:
                    parts.append(f"Context: {item['context'][:200]}...")
                parts.append(f"Response: {item['response']}")
                if item['evaluation_summary']:
                    parts.append(f"Eval: {item['evaluation_summary']}")
                parts.append("---")

        return "\n".join(parts)

    def _check_compaction(self) -> None:
        """Check if compaction is needed and trigger if so."""
        should_compact = False

        # Check item count
        if len(self.history) > self.max_history_items:
            should_compact = True

        # Check character count (approximate)
        current_chars = sum(len(str(item)) for item in self.history)
        if current_chars > self.max_context_chars:
            should_compact = True

        if should_compact:
            self._compact_history()

    def _compact_history(self) -> None:
        """
        Compact older history items into a summary using LLM.
        Keeps the most recent N items as raw history.
        """
        keep_count = self.keep_recent_items
        if len(self.history) <= keep_count:
            return

        # Items to compact
        to_compact = self.history[:-keep_count] if keep_count > 0 else self.history
        
        # Format for summarization
        text_to_summarize = "\n\n".join([
            f"Interaction {i}:\nContext: {item['context']}\nResponse: {item['response']}\nEval Result: {item['evaluation_summary']}"
            for i, item in enumerate(to_compact)
        ])

        try:
            # Call LLM to summarize
            # We use a synchronous call here since this runs during the tool execution
            # which is already in a thread or async context depending on the server
            # Using the settings model or a specific cheap model for summarization
            result = self.client.messages.create(
                model=settings.evaluator.compaction_model,
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": f"""Summarize this interaction history effectively. Preserve key decisions, tool results, and evaluation warnings. The summary should be concise (max {self.compaction_target_chars} chars).

History to summarize:
{text_to_summarize}

Concise summary:"""
                }]
            )

            new_summary = result.content[0].text

            # Merge with existing compacted history
            if self.compacted_history:
                merge_result = self.client.messages.create(
                    model=settings.evaluator.compaction_model,
                    max_tokens=800,
                    messages=[{
                        "role": "user",
                        "content": f"""Merge these two session summaries into one concise summary (max {self.compaction_target_chars} chars):

Previous Summary:
{self.compacted_history}

New Summary:
{new_summary}

Merged summary:"""
                    }]
                )
                self.compacted_history = merge_result.content[0].text
            else:
                self.compacted_history = new_summary

            # Remove compacted items, keep recent
            self.history = (
                self.history[-keep_count:]
                if keep_count > 0
                else []
            )
            self.context_version += 1

        except Exception:
            # If compaction fails, just trim oldest items
            self.history = self.history[-self.max_history_items // 2:]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current context state.

        Returns:
            Dictionary with context statistics.
        """
        return {
            "history_items": len(self.history),
            "compacted_history_chars": len(self.compacted_history),
            "context_version": self.context_version,
        }

    def clear(self) -> None:
        """Clear all context."""
        self.history = []
        self.compacted_history = ""
        self.context_version = 0

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the current history list."""
        return self.history.copy()
