"""
Helper functions for the AI Evaluator Framework.
"""

from typing import Any, Dict, List

from ..config.settings import settings


def get_risk_level(score: float) -> str:
    """
    Calculate risk level from score.

    Args:
        score: The overall score (0.0 to 1.0).

    Returns:
        Risk level string: "low", "medium", or "high".
    """
    if score >= 0.8:
        return "low"
    elif score >= 0.5:
        return "medium"
    else:
        return "high"


def collect_warnings(evaluation: Dict[str, Any]) -> List[str]:
    """
    Collect all warnings from evaluation results.

    Args:
        evaluation: The evaluation result dictionary.

    Returns:
        List of warning messages.
    """
    warnings = []

    if evaluation.get("hallucination_score", 1.0) < 0.5:
        hallucination_count = len(evaluation.get("hallucinations", []))
        warnings.append(
            f"High hallucination risk: {hallucination_count} potential hallucination(s)"
        )

    if evaluation.get("tool_consistency_score", 1.0) < 0.7:
        mismatch_count = len(evaluation.get("tool_mismatches", []))
        warnings.append(
            f"Tool consistency issues: {mismatch_count} mismatch(es)"
        )

    if evaluation.get("context_consistency_score", 1.0) < 0.7:
        contradiction_count = len(evaluation.get("contradictions", []))
        warnings.append(
            f"Context contradictions: {contradiction_count} found"
        )

    overconfident_count = len(evaluation.get("overconfident_claims", []))
    if overconfident_count > 2:
        warnings.append(
            f"Overconfidence: {overconfident_count} unsupported confident claim(s)"
        )

    return warnings


def generate_suggestions(
    evaluation: Dict[str, Any],
    judge_result: Dict[str, Any]
) -> List[str]:
    """
    Generate actionable suggestions based on evaluation.

    Args:
        evaluation: The evaluation result dictionary.
        judge_result: The Petri judge result dictionary.

    Returns:
        List of suggestion messages.
    """
    suggestions = []

    # Hallucination suggestions
    if evaluation.get("hallucinations"):
        suggestions.append(
            "Add evidence/sources for claims or use more cautious language"
        )
        suggestions.append(
            "Consider adding tool calls to verify facts before stating them"
        )

    # Tool consistency suggestions
    tool_mismatches = evaluation.get("tool_mismatches", [])
    for mismatch in tool_mismatches:
        tool_name = mismatch.get("tool", "unknown tool")
        suggestions.append(
            f"Actually call {tool_name} before claiming results from it"
        )

    # Context consistency suggestions
    if evaluation.get("contradictions"):
        suggestions.append(
            "Review conversation history to ensure consistency"
        )

    # Judge-based suggestions
    judge_score = judge_result.get("score", 1.0)
    if judge_score < settings.evaluator.pass_threshold:
        summary = judge_result.get("summary", "Quality issues detected")
        suggestions.append(f"Judge identified issues: {summary}")

    # Add recommendations from judge
    recommendations = judge_result.get("recommendations", [])
    for rec in recommendations[:3]:  # Limit to 3
        suggestions.append(rec)

    return suggestions


def generate_session_recommendations(
    evaluations: List[Dict[str, Any]]
) -> List[str]:
    """
    Generate session-level recommendations.

    Args:
        evaluations: List of evaluation results from the session.

    Returns:
        List of recommendation messages.
    """
    if not evaluations:
        return []

    recommendations = []

    # Calculate average hallucination score (lower = more hallucinations)
    avg_hallucination_score = sum(
        e.get("dimensions", {}).get("hallucination", {}).get("score", 1.0)
        for e in evaluations
    ) / len(evaluations)

    if avg_hallucination_score < 0.6:
        recommendations.append(
            "High hallucination rate detected - consider stricter system prompts or different model"
        )

    # Tool consistency score
    avg_tool_score = sum(
        e.get("dimensions", {}).get("tool_consistency", {}).get("score", 1.0)
        for e in evaluations
    ) / len(evaluations)

    if avg_tool_score < 0.7:
        recommendations.append(
            "Frequent tool consistency issues - review tool calling logic"
        )

    # Petri evaluation critical issues
    total_critical = sum(
        len(e.get("dimensions", {}).get("petri_evaluation", {}).get("critical_issues", []))
        for e in evaluations
    )
    if total_critical > 0:
        recommendations.append(
            f"Found {total_critical} critical issue(s) across evaluations - review petri assessment"
        )

    # Average score
    avg_score = sum(
        e.get("overall_score", 0)
        for e in evaluations
    ) / len(evaluations)

    if avg_score < 0.6:
        recommendations.append(
            "Overall quality below threshold - consider prompt refinement or model change"
        )

    return recommendations


def format_evaluation_result(
    evaluation: Dict[str, Any],
    verbose: bool = False
) -> str:
    """
    Format evaluation result for display.

    Args:
        evaluation: The evaluation result dictionary.
        verbose: Whether to include detailed information.

    Returns:
        Formatted string representation.
    """
    lines = [
        f"Overall Score: {evaluation.get('overall_score', 0):.3f}",
        f"Risk Level: {evaluation.get('risk_level', 'unknown')}",
        f"Pass: {'Yes' if evaluation.get('pass') else 'No'}",
    ]

    if verbose:
        lines.extend([
            "",
            "Dimension Scores:",
            f"  Hallucination: {evaluation.get('hallucination_score', 0):.3f}",
            f"  Tool Consistency: {evaluation.get('tool_consistency_score', 0):.3f}",
            f"  Context Consistency: {evaluation.get('context_consistency_score', 0):.3f}",
            f"  Confidence: {evaluation.get('confidence_score', 0):.3f}",
        ])

        warnings = evaluation.get("warnings", [])
        if warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in warnings:
                lines.append(f"  - {warning}")

        suggestions = evaluation.get("suggestions", [])
        if suggestions:
            lines.append("")
            lines.append("Suggestions:")
            for suggestion in suggestions:
                lines.append(f"  - {suggestion}")

    return "\n".join(lines)
