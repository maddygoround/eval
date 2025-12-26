"""
Eval

A comprehensive framework for evaluating AI responses using Inspect AI
and Petri-style alignment patterns. Uses a unified scorer that evaluates
all dimensions in a single LLM call.
"""

__version__ = "0.4.0"
__author__ = "Eval Team"

from .tasks.eval import Evaluator
from .scorers.judge import unified_alignment_scorer

__all__ = [
    "Evaluator",
    "unified_alignment_scorer",
]
