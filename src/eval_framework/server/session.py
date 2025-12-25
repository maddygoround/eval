"""
Session State Management for MCP Server.

This module manages the current session state including
context accumulation and evaluation tracking.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..utils.context import ContextManager
from ..utils.storage import EvaluationStorage
from ..core.evaluator import ResponseEvaluator
from ..core.judge import PetriJudge


@dataclass
class SessionState:
    """
    Holds the current session state.

    This is a singleton-like structure that maintains state
    across MCP tool calls.
    """
    id: Optional[str] = None
    name: str = ""
    description: str = ""
    started_at: Optional[str] = None
    evaluations: List[Dict[str, Any]] = field(default_factory=list)

    # Components
    context_manager: ContextManager = field(default_factory=ContextManager)
    storage: EvaluationStorage = field(default_factory=EvaluationStorage)
    evaluator: ResponseEvaluator = field(default_factory=ResponseEvaluator)
    judge: PetriJudge = field(default_factory=PetriJudge)

    def start_session(
        self,
        session_id: str,
        name: str = "",
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Start a new evaluation session.

        Args:
            session_id: Unique identifier for the session.
            name: Human-readable session name.
            description: Description of what's being tested.

        Returns:
            Session info dictionary.
        """
        self.id = session_id
        self.name = name or session_id
        self.description = description
        self.started_at = datetime.now().isoformat()
        self.evaluations = []
        self.context_manager.clear()

        # Create session in storage
        self.storage.create_session(session_id, name, description)

        return {
            "session_id": session_id,
            "name": self.name,
            "message": f"Started new evaluation session: {self.name}"
        }

    def add_evaluation(self, evaluation: Dict[str, Any]) -> None:
        """Add an evaluation to the current session."""
        self.evaluations.append(evaluation)
        if self.id:
            self.storage.store(self.id, evaluation)

    def get_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        context_stats = self.context_manager.get_stats()
        return {
            "session_id": self.id,
            "name": self.name,
            "started_at": self.started_at,
            "total_evaluations": len(self.evaluations),
            "context": context_stats
        }

    def clear_context(self) -> None:
        """Clear context while preserving session."""
        self.context_manager.clear()


# Global session state instance
current_session = SessionState()


def get_session() -> SessionState:
    """Get the current session state."""
    return current_session


def reset_session() -> SessionState:
    """Reset the session state."""
    global current_session
    current_session = SessionState()
    return current_session
