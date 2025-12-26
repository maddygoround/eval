"""
Helper functions for the AI Evaluator Framework.
"""

from typing import Any, Dict, List

from .config.settings import settings


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


def generate_session_recommendations(session_info: Dict[str, Any]) -> List[str]:
    """
    Generate recommendations based on session history.

    Args:
        session_info: The session information dictionary.

    Returns:
        List of recommendation strings.
    """
    recommendations = []
    evaluations = session_info.get("evaluations", [])

    if not evaluations:
        return recommendations

    # Check trends
    total_critical = sum(
        len(e.get("petri_eval", {}).get("critical_issues", []))
        for e in evaluations
    )
    if total_critical > 3:
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
