"""
Data models for the AI Evaluator Framework.
"""

from .evaluation import (
    EvaluationResult,
    DimensionScore,
    HallucinationIssue,
    ToolIssue,
    Contradiction,
    OverconfidentClaim,
    JudgeDimension,
    JudgeResult,
    SessionInfo,
    Interaction,
)

__all__ = [
    "EvaluationResult",
    "DimensionScore",
    "HallucinationIssue",
    "ToolIssue",
    "Contradiction",
    "OverconfidentClaim",
    "JudgeDimension",
    "JudgeResult",
    "SessionInfo",
    "Interaction",
]
