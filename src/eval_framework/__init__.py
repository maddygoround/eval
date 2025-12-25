"""
AI Evaluator Framework
A comprehensive framework for evaluating AI responses using Inspect AI and Petri patterns.
"""

__version__ = "0.1.0"
__author__ = "AI Evaluator Team"

from .core.evaluator import ResponseEvaluator
from .core.judge import PetriJudge
from .core.scorers import (
    hallucination_scorer,
    tool_consistency_scorer,
    petri_scorer
)

__all__ = [
    "ResponseEvaluator",
    "PetriJudge",
    "hallucination_scorer",
    "tool_consistency_scorer",
    "petri_scorer",
]
