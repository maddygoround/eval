"""
Data models for evaluation results and related structures.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class HallucinationIssue:
    """Represents a detected hallucination."""
    claim: str
    reason: str
    confidence: float


@dataclass
class ToolIssue:
    """Represents a tool consistency issue."""
    type: str  # e.g., "mentioned_not_used"
    tool: str
    claim: str
    severity: str  # low/medium/high


@dataclass
class Contradiction:
    """Represents a context contradiction."""
    contradiction: str
    context_quote: str
    response_quote: str
    severity: str


@dataclass
class OverconfidentClaim:
    """Represents an overconfident claim."""
    claim: str
    context: str
    category: str
    suggestion: str


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""
    name: str
    score: float
    evidence: str = ""
    issues: List[str] = field(default_factory=list)


@dataclass
class JudgeDimension:
    """A dimension to evaluate on (from Petri)."""
    name: str
    description: str
    low_score_indicator: str
    high_score_indicator: str


@dataclass
class JudgeResult:
    """Result from Petri-style judge evaluation."""
    dimensions: List[DimensionScore]
    aggregate_score: float
    critical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    summary: str = ""

    @property
    def score(self) -> float:
        """Alias for aggregate_score."""
        return self.aggregate_score


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    timestamp: str
    model: str
    overall_score: float
    risk_level: str
    hallucination_score: float
    tool_consistency_score: float
    context_consistency_score: float
    confidence_score: float
    petri_score: float
    hallucinations: List[Dict[str, Any]] = field(default_factory=list)
    tool_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    overconfident_claims: List[Dict[str, Any]] = field(default_factory=list)
    petri_eval: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "overall_score": self.overall_score,
            "risk_level": self.risk_level,
            "hallucination_score": self.hallucination_score,
            "tool_consistency_score": self.tool_consistency_score,
            "context_consistency_score": self.context_consistency_score,
            "confidence_score": self.confidence_score,
            "petri_score": self.petri_score,
            "hallucinations": self.hallucinations,
            "tool_mismatches": self.tool_mismatches,
            "contradictions": self.contradictions,
            "overconfident_claims": self.overconfident_claims,
            "petri_eval": self.petri_eval,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "pass": self.passed,
        }


@dataclass
class Interaction:
    """Represents a single interaction in session history."""
    timestamp: str
    context: str
    response: str
    evaluation_summary: str = ""
    index: int = 0


@dataclass
class SessionInfo:
    """Session tracking information."""
    id: Optional[str] = None
    name: str = ""
    description: str = ""
    started_at: Optional[str] = None
    history: List[Interaction] = field(default_factory=list)
    compacted_history: str = ""
    evaluations: List[EvaluationResult] = field(default_factory=list)
    context_version: int = 0

    def clear_context(self):
        """Clear context while preserving session."""
        self.history = []
        self.compacted_history = ""
        self.context_version = 0
