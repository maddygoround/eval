"""
MCP Tool Handlers for AI Evaluator.

This module implements the handler functions for each MCP tool.
"""

from typing import Any, Dict
from datetime import datetime

from .session import get_session
from ..utils import (
    get_risk_level,
    generate_session_recommendations
)
from ..config.settings import settings


def evaluate_response_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main evaluation handler using Inspect AI + Petri patterns.

    Args:
        args: Tool arguments including response, context, tools info.

    Returns:
        Comprehensive evaluation result.
    """
    session = get_session()

    response = args["response"]
    provided_context = args.get("context", "")
    tools_available = args.get("tools_available", [])
    tools_used = args.get("tools_used", [])
    import os
    # Priority: 1. ENV/Settings (JUDGE_MODEL) 2. Tool Argument
    env_model = os.getenv("JUDGE_MODEL") or settings.evaluator.judge_model
    arg_model = args.get("model")
    
    if env_model and env_model != "unknown":
        model = env_model
    elif arg_model and arg_model != "unknown":
        model = arg_model
    else:
        model = "unknown"
    modified_files = args.get("modified_files", [])
    tool_call_log = args.get("tool_call_log", [])
    use_accumulated = args.get("use_accumulated_context", True)

    # Build full context: accumulated session context + provided context
    if use_accumulated:
        accumulated = session.context_manager.get_accumulated_context()
        if accumulated and provided_context:
            full_context = f"{accumulated}\n\n[Current Context]\n{provided_context}"
        elif accumulated:
            full_context = accumulated
        else:
            full_context = provided_context
    else:
        full_context = provided_context

    # Run comprehensive evaluation with full context
    evaluation = session.evaluator.evaluate_comprehensive(
        response=response,
        context=full_context,
        tools_available=tools_available,
        tools_used=tools_used,
        model=model,
        modified_files=modified_files,
        tool_call_log=tool_call_log
    )

    # Extract scores from unified evaluation results
    hallucination_score = evaluation.get("hallucination_score", 1.0)
    tool_consistency_score = evaluation.get("tool_consistency_score", 1.0)
    petri_score = evaluation.get("petri_score", 1.0)

    # Use the pre-calculated overall score from evaluator, or compute if missing
    overall_score = evaluation.get("overall_score",
        (hallucination_score + tool_consistency_score + petri_score) / 3
    )

    # Extract petri_eval nested data
    petri_eval = evaluation.get("petri_eval", {})
    petri_dimensions = petri_eval.get("dimensions", [])
    critical_issues = petri_eval.get("critical_issues", []) or evaluation.get("warnings", [])
    recommendations = petri_eval.get("recommendations", []) or evaluation.get("suggestions", [])
    petri_evidence = petri_eval.get("evidence", {})

    # Build result from unified evaluation
    result = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "overall_score": round(overall_score, 3),
        "risk_level": get_risk_level(overall_score),
        "dimensions": {
            "hallucination": {
                "score": hallucination_score,
                "explanation": petri_evidence.get("hallucination", {}).get("score_rationale", "")
            },
            "tool_consistency": {
                "score": tool_consistency_score,
                "explanation": petri_evidence.get("tool_consistency", {}).get("score_rationale", "")
            },
            "petri_evaluation": {
                "score": petri_score,
                "explanation": petri_eval.get("summary", ""),
                "dimensions": petri_dimensions,
                "raw_scores": petri_eval.get("raw_scores", {}),
                "critical_issues": critical_issues,
                "recommendations": recommendations
            }
        },
        "warnings": critical_issues,
        "suggestions": recommendations,
        "pass": overall_score >= settings.evaluator.pass_threshold
    }

    # Store evaluation
    session.add_evaluation(result)

    # Add this interaction to context history for future evaluations
    eval_summary = f"Score: {result['overall_score']:.2f}, Risk: {result['risk_level']}"
    if result["warnings"]:
        eval_summary += f", Warnings: {len(result['warnings'])}"
    session.context_manager.add_interaction(
        context=provided_context,
        response=response,
        evaluation_summary=eval_summary
    )

    # Add context stats to result
    result["context_stats"] = session.context_manager.get_stats()

    return result


def check_hallucinations_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quick hallucination check using unified evaluation.

    Args:
        args: Tool arguments including response and context.

    Returns:
        Hallucination check results extracted from unified evaluation.
    """
    session = get_session()

    response = args["response"]
    context = args.get("context", "")

    # Run unified evaluation
    evaluation = session.evaluator.evaluate_comprehensive(
        response=response,
        context=context
    )

    # Extract hallucination-specific data
    petri_eval = evaluation.get("petri_eval", {})
    evidence = petri_eval.get("evidence", {})
    hallucination_evidence = evidence.get("hallucination", {})
    hallucination_issues = hallucination_evidence.get("issues", [])
    hallucination_score = evaluation.get("hallucination_score", 1.0)

    return {
        "hallucination_score": hallucination_score,
        "hallucinations_found": len(hallucination_issues),
        "hallucinations": hallucination_issues,
        "explanation": hallucination_evidence.get("score_rationale", ""),
        "risk_level": (
            "high" if hallucination_score < 0.5
            else "medium" if hallucination_score < 0.7
            else "low"
        ),
        "summary": f"Hallucination score: {hallucination_score:.2f} - Found {len(hallucination_issues)} issue(s)"
    }


def verify_tool_consistency_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify tool usage consistency using unified evaluation.

    Args:
        args: Tool arguments including response, tools_available, tools_used.

    Returns:
        Tool consistency check results extracted from unified evaluation.
    """
    session = get_session()

    response = args["response"]
    tools_available = args.get("tools_available", [])
    tools_used = args.get("tools_used", [])
    context = args.get("context", "")

    # Run unified evaluation
    evaluation = session.evaluator.evaluate_comprehensive(
        response=response,
        context=context,
        tools_available=tools_available,
        tools_used=tools_used
    )

    # Extract tool consistency data
    petri_eval = evaluation.get("petri_eval", {})
    evidence = petri_eval.get("evidence", {})
    tool_evidence = evidence.get("tool_consistency", {})
    tool_issues = tool_evidence.get("issues", [])
    tool_score = evaluation.get("tool_consistency_score", 1.0)

    return {
        "tool_consistency_score": tool_score,
        "consistent": tool_score >= 0.7,
        "issues": tool_issues,
        "explanation": tool_evidence.get("score_rationale", ""),
        "summary": f"Tool consistency score: {tool_score:.2f} - Found {len(tool_issues)} issue(s)"
    }


def compare_models_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare multiple model responses handler.

    Args:
        args: Tool arguments including responses list and context.

    Returns:
        Comparison results with rankings.
    """
    session = get_session()

    responses = args["responses"]
    context = args.get("context", "")

    results = []
    for item in responses:
        eval_result = session.evaluator.evaluate_comprehensive(
            response=item["response"],
            context=context
        )
        # Calculate overall score from the three dimensions
        hallucination_score = eval_result.get("hallucination_score", 1.0)
        tool_score = eval_result.get("tool_consistency_score", 1.0)
        petri_score = eval_result.get("petri_score", 1.0)
        overall = (hallucination_score + tool_score + petri_score) / 3

        results.append({
            "model": item["model"],
            "score": round(overall, 3),
            "hallucination_score": hallucination_score,
            "tool_consistency_score": tool_score,
            "petri_score": petri_score
        })

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "comparison": results,
        "best_model": results[0]["model"],
        "recommendation": f"{results[0]['model']} performed best with score {results[0]['score']:.2f}"
    }


def session_report_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate session report handler.

    Args:
        args: Tool arguments including session_id and detailed flag.

    Returns:
        Session statistics and recommendations.
    """
    session = get_session()

    session_id = args.get("session_id", session.id)
    detailed = args.get("detailed", False)

    evaluations = session.evaluations

    if not evaluations:
        return {
            "session_id": session_id,
            "total_evaluations": 0,
            "message": "No evaluations in this session yet"
        }

    # Calculate statistics
    total = len(evaluations)
    avg_score = sum(e["overall_score"] for e in evaluations) / total

    # Calculate average scores by dimension
    avg_hallucination = sum(
        e["dimensions"]["hallucination"]["score"] for e in evaluations
    ) / total
    avg_tool_consistency = sum(
        e["dimensions"]["tool_consistency"]["score"] for e in evaluations
    ) / total
    avg_petri = sum(
        e["dimensions"]["petri_evaluation"]["score"] for e in evaluations
    ) / total

    # Count critical issues from petri evaluations
    total_critical_issues = sum(
        len(e["dimensions"]["petri_evaluation"].get("critical_issues", []))
        for e in evaluations
    )

    report = {
        "session_id": session_id,
        "total_evaluations": total,
        "average_score": round(avg_score, 3),
        "pass_rate": sum(1 for e in evaluations if e["pass"]) / total,
        "dimension_averages": {
            "hallucination": round(avg_hallucination, 3),
            "tool_consistency": round(avg_tool_consistency, 3),
            "petri_evaluation": round(avg_petri, 3)
        },
        "total_critical_issues": total_critical_issues,
        "risk_distribution": {
            "high": sum(1 for e in evaluations if e["risk_level"] == "high"),
            "medium": sum(1 for e in evaluations if e["risk_level"] == "medium"),
            "low": sum(1 for e in evaluations if e["risk_level"] == "low")
        },
        "recommendations": generate_session_recommendations(evaluations)
    }

    if detailed:
        report["detailed_issues"] = [
            {
                "index": i,
                "score": e["overall_score"],
                "warnings": e.get("warnings", []),
                "critical_issues": e["dimensions"]["petri_evaluation"].get("critical_issues", [])
            }
            for i, e in enumerate(evaluations) if not e["pass"]
        ]

    return report


def start_session_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Start new evaluation session handler.

    Args:
        args: Tool arguments including name and description.

    Returns:
        New session info.
    """
    session = get_session()

    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    name = args.get("name", session_id)
    description = args.get("description", "")

    return session.start_session(session_id, name, description)


def get_context_stats_handler(_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get context accumulation statistics handler.

    Returns:
        Context statistics and preview.
    """
    session = get_session()

    stats = session.context_manager.get_stats()
    stats["session_id"] = session.id
    stats["total_evaluations"] = len(session.evaluations)

    accumulated = session.context_manager.get_accumulated_context()
    stats["accumulated_context_preview"] = (
        accumulated[:500] + "..."
        if len(accumulated) > 500
        else accumulated
    )

    return stats


def clear_context_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clear accumulated context handler.

    Args:
        args: Tool arguments including confirm flag.

    Returns:
        Clear status and updated stats.
    """
    session = get_session()

    confirm = args.get("confirm", False)

    if not confirm:
        return {
            "cleared": False,
            "message": "Set confirm=true to clear context"
        }

    session.clear_context()

    return {
        "cleared": True,
        "message": "Context cleared successfully",
        "context_stats": session.context_manager.get_stats()
    }
