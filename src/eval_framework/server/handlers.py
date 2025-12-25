"""
MCP Tool Handlers for AI Evaluator.

This module implements the handler functions for each MCP tool.
"""

from typing import Any, Dict, List
from datetime import datetime

from .session import get_session
from ..utils.helpers import (
    get_risk_level,
    collect_warnings,
    generate_suggestions,
    generate_session_recommendations
)
from ..config.settings import settings


async def evaluate_response_handler(args: Dict[str, Any]) -> Dict[str, Any]:
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
    model = args.get("model", "unknown")
    modified_files = args.get("modified_files", [])
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
    evaluation = await session.evaluator.evaluate_comprehensive(
        response=response,
        context=full_context,
        tools_available=tools_available,
        tools_used=tools_used,
        modified_files=modified_files
    )

    # Use Petri-style judge for behavioral assessment
    judge_result = await session.judge.evaluate(
        context=full_context,
        response=response,
        tools_available=tools_available,
        tools_used=tools_used
    )

    # Combine results
    result = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "overall_score": evaluation["overall_score"],
        "risk_level": get_risk_level(evaluation["overall_score"]),
        "dimensions": {
            "hallucination": {
                "score": evaluation["hallucination_score"],
                "issues": evaluation["hallucinations"],
                "count": len(evaluation["hallucinations"])
            },
            "tool_consistency": {
                "score": evaluation["tool_consistency_score"],
                "issues": evaluation["tool_mismatches"]
            },
            "context_consistency": {
                "score": evaluation["context_consistency_score"],
                "contradictions": evaluation["contradictions"]
            },
            "confidence_calibration": {
                "score": evaluation["confidence_score"],
                "overconfident_claims": evaluation["overconfident_claims"]
            }
        },
        "judge_assessment": judge_result,
        "warnings": collect_warnings(evaluation),
        "suggestions": generate_suggestions(evaluation, judge_result),
        "pass": evaluation["overall_score"] >= settings.evaluator.pass_threshold
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


async def check_hallucinations_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quick hallucination check handler.

    Args:
        args: Tool arguments including response and strict_mode.

    Returns:
        Hallucination check results.
    """
    session = get_session()

    response = args["response"]
    strict_mode = args.get("strict_mode", False)

    hallucinations = await session.evaluator.detect_hallucinations(
        response=response,
        strict=strict_mode
    )

    return {
        "hallucinations_found": len(hallucinations),
        "hallucinations": hallucinations,
        "risk_level": (
            "high" if len(hallucinations) > 3
            else "medium" if len(hallucinations) > 0
            else "low"
        ),
        "summary": f"Found {len(hallucinations)} potential hallucination(s)"
    }


async def verify_tool_consistency_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify tool usage consistency handler.

    Args:
        args: Tool arguments including response, tools_available, tools_used.

    Returns:
        Tool consistency check results.
    """
    session = get_session()

    response = args["response"]
    tools_available = args["tools_available"]
    tools_used = args["tools_used"]

    issues = await session.evaluator.verify_tool_consistency(
        response=response,
        tools_available=tools_available,
        tools_used=tools_used
    )

    return {
        "consistent": len(issues) == 0,
        "issues": issues,
        "tools_mentioned_not_used": [
            i["tool"] for i in issues if i["type"] == "mentioned_not_used"
        ],
        "summary": f"Found {len(issues)} tool consistency issue(s)"
    }


async def compare_models_handler(args: Dict[str, Any]) -> Dict[str, Any]:
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
        eval_result = await session.evaluator.evaluate_comprehensive(
            response=item["response"],
            context=context
        )
        results.append({
            "model": item["model"],
            "score": eval_result["overall_score"],
            "hallucination_count": len(eval_result["hallucinations"]),
            "consistency_score": eval_result["context_consistency_score"]
        })

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "comparison": results,
        "best_model": results[0]["model"],
        "recommendation": f"{results[0]['model']} performed best with score {results[0]['score']:.2f}"
    }


async def session_report_handler(args: Dict[str, Any]) -> Dict[str, Any]:
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
    total_hallucinations = sum(
        e["dimensions"]["hallucination"]["count"] for e in evaluations
    )
    avg_score = sum(e["overall_score"] for e in evaluations) / total

    # Count issues by type
    issues_summary = {
        "hallucinations": total_hallucinations,
        "tool_mismatches": sum(
            len(e["dimensions"]["tool_consistency"]["issues"])
            for e in evaluations
        ),
        "contradictions": sum(
            len(e["dimensions"]["context_consistency"]["contradictions"])
            for e in evaluations
        ),
        "overconfident_claims": sum(
            len(e["dimensions"]["confidence_calibration"]["overconfident_claims"])
            for e in evaluations
        )
    }

    report = {
        "session_id": session_id,
        "total_evaluations": total,
        "average_score": round(avg_score, 3),
        "pass_rate": sum(1 for e in evaluations if e["pass"]) / total,
        "issues_summary": issues_summary,
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
                "warnings": e["warnings"]
            }
            for i, e in enumerate(evaluations) if not e["pass"]
        ]

    return report


async def start_session_handler(args: Dict[str, Any]) -> Dict[str, Any]:
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
