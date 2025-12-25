"""
Context Management with Automatic Compaction.

This module provides session context management with automatic
compaction to prevent unbounded context growth.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import anthropic

from ..config.settings import settings
from ..models.evaluation import Interaction


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

    def add_interaction(
        self,
        context: str,
        response: str,
        evaluation_summary: str = ""
    ) -> None:
        """
        Add a new interaction to the session history.

        Args:
            context: The context for this interaction.
            response: The response generated.
            evaluation_summary: Summary of evaluation results.
        """
        # Truncate long responses to save space
        truncated_response = (
            response[:500] + "..."
            if len(response) > 500
            else response
        )

        interaction = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "response": truncated_response,
            "evaluation_summary": evaluation_summary,
            "index": len(self.history)
        }
        self.history.append(interaction)

        # Check if compaction is needed
        if self._needs_compaction():
            self._compact_history()

    def get_accumulated_context(self) -> str:
        """
        Get the full accumulated context for evaluation.

        Combines compacted history summary with recent interactions.

        Returns:
            Combined context string.
        """
        parts = []

        # Add compacted history summary if exists
        if self.compacted_history:
            parts.append(
                f"[Previous Session Context Summary]\n{self.compacted_history}\n"
            )

        # Add recent history items
        recent_items = self.history[-self.keep_recent_items:]
        if recent_items:
            parts.append("[Recent Interactions]")
            for item in recent_items:
                context_preview = item['context'][:200] + "..." if len(item['context']) > 200 else item['context']
                response_preview = item['response'][:200] + "..." if len(item['response']) > 200 else item['response']
                parts.append(f"- Context: {context_preview}")
                parts.append(f"  Response: {response_preview}")
                if item.get("evaluation_summary"):
                    parts.append(f"  Eval: {item['evaluation_summary']}")

        return "\n".join(parts)

    def _needs_compaction(self) -> bool:
        """Check if context needs compaction."""
        history_size = len(self.history)
        context_chars = sum(
            len(h.get("context", "")) + len(h.get("response", ""))
            for h in self.history
        )

        return (
            history_size > self.max_history_items or
            context_chars > self.max_context_chars
        )

    def _compact_history(self) -> None:
        """Compact older history into a summary using LLM."""
        # Keep recent items separate
        keep_count = self.keep_recent_items
        items_to_compact = (
            self.history[:-keep_count]
            if keep_count > 0
            else self.history
        )

        if not items_to_compact:
            return

        # Build text to summarize
        history_text = "\n\n".join([
            f"Interaction {i+1}:\n- Context: {item['context']}\n- Response: {item['response']}\n- Eval: {item.get('evaluation_summary', 'N/A')}"
            for i, item in enumerate(items_to_compact)
        ])

        try:
            # Create summary using LLM
            result = self.client.messages.create(
                model=settings.evaluator.compaction_model,
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": f"""Summarize this conversation history concisely, preserving:
1. Key topics discussed
2. Important decisions or facts established
3. Any issues or patterns identified in evaluations
4. Tool usage patterns

Keep the summary under {self.compaction_target_chars} characters.

History to summarize:
{history_text}

Provide a concise summary:"""
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
