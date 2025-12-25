"""
AI Response Evaluator MCP Server
Real-time evaluation of AI responses using Inspect AI and Petri patterns
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
import mcp.server.stdio
import anthropic

from evaluator import ResponseEvaluator
from judge import PetriJudge
from storage import EvaluationStorage

# Initialize MCP server
app = Server("ai-evaluator")

# Initialize components
evaluator = ResponseEvaluator()
judge = PetriJudge()
storage = EvaluationStorage()
compaction_client = anthropic.Anthropic()

# Context management configuration
CONTEXT_CONFIG = {
    "max_history_items": 20,          # Max items before compaction triggers
    "max_context_chars": 15000,       # Max chars before compaction
    "compaction_target_chars": 5000,  # Target size after compaction
    "keep_recent_items": 3            # Always keep N most recent items uncompacted
}

# Session tracking with context accumulation
current_session = {
    "id": None,
    "history": [],              # Full history of context/response pairs
    "compacted_history": "",    # Compacted summary of older history
    "evaluations": [],
    "context_version": 0        # Increments on each compaction
}


class ContextManager:
    """
    Manages session context with automatic compaction.
    Accumulates context across the session and compacts when needed.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or CONTEXT_CONFIG
        self.client = compaction_client

    def add_interaction(self, context: str, response: str, evaluation_summary: str = ""):
        """Add a new interaction to the session history"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "response": response[:500] + "..." if len(response) > 500 else response,  # Truncate long responses
            "evaluation_summary": evaluation_summary,
            "index": len(current_session["history"])
        }
        current_session["history"].append(interaction)

        # Check if compaction is needed
        if self._needs_compaction():
            self._compact_history()

    def get_accumulated_context(self) -> str:
        """
        Get the full accumulated context for evaluation.
        Combines compacted history with recent interactions.
        """
        parts = []

        # Add compacted history summary if exists
        if current_session["compacted_history"]:
            parts.append(f"[Previous Session Context Summary]\n{current_session['compacted_history']}\n")

        # Add recent history items
        recent_items = current_session["history"][-self.config["keep_recent_items"]:]
        if recent_items:
            parts.append("[Recent Interactions]")
            for item in recent_items:
                parts.append(f"- Context: {item['context'][:200]}...")
                parts.append(f"  Response: {item['response'][:200]}...")
                if item.get("evaluation_summary"):
                    parts.append(f"  Eval: {item['evaluation_summary']}")

        return "\n".join(parts)

    def _needs_compaction(self) -> bool:
        """Check if context needs compaction"""
        history_size = len(current_session["history"])
        context_chars = sum(
            len(h.get("context", "")) + len(h.get("response", ""))
            for h in current_session["history"]
        )

        return (
            history_size > self.config["max_history_items"] or
            context_chars > self.config["max_context_chars"]
        )

    def _compact_history(self):
        """Compact older history into a summary"""
        # Keep recent items separate
        keep_count = self.config["keep_recent_items"]
        items_to_compact = current_session["history"][:-keep_count] if keep_count > 0 else current_session["history"]

        if not items_to_compact:
            return

        # Build text to summarize
        history_text = "\n\n".join([
            f"Interaction {i+1}:\n- Context: {item['context']}\n- Response: {item['response']}\n- Eval: {item.get('evaluation_summary', 'N/A')}"
            for i, item in enumerate(items_to_compact)
        ])

        # Use LLM to create summary
        try:
            result = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": f"""Summarize this conversation history concisely, preserving:
1. Key topics discussed
2. Important decisions or facts established
3. Any issues or patterns identified in evaluations
4. Tool usage patterns

Keep the summary under {self.config['compaction_target_chars']} characters.

History to summarize:
{history_text}

Provide a concise summary:"""
                }]
            )

            new_summary = result.content[0].text

            # Merge with existing compacted history
            if current_session["compacted_history"]:
                # Further compress if combining
                merge_result = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=800,
                    messages=[{
                        "role": "user",
                        "content": f"""Merge these two session summaries into one concise summary (max {self.config['compaction_target_chars']} chars):

Previous Summary:
{current_session['compacted_history']}

New Summary:
{new_summary}

Merged summary:"""
                    }]
                )
                current_session["compacted_history"] = merge_result.content[0].text
            else:
                current_session["compacted_history"] = new_summary

            # Remove compacted items, keep recent
            current_session["history"] = current_session["history"][-keep_count:] if keep_count > 0 else []
            current_session["context_version"] += 1

        except Exception as e:
            # If compaction fails, just trim oldest items
            current_session["history"] = current_session["history"][-self.config["max_history_items"]//2:]

    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about current context state"""
        return {
            "history_items": len(current_session["history"]),
            "compacted_history_chars": len(current_session["compacted_history"]),
            "context_version": current_session["context_version"],
            "total_evaluations": len(current_session["evaluations"])
        }

    def clear(self):
        """Clear all context"""
        current_session["history"] = []
        current_session["compacted_history"] = ""
        current_session["context_version"] = 0


# Initialize context manager
context_manager = ContextManager()


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available evaluation tools"""
    return [
        Tool(
            name="evaluate_response",
            description="""Evaluate an AI response for hallucinations, consistency, and quality.
            
            Use this during development to check if an AI response has issues like:
            - Hallucinations (unfounded claims)
            - Tool call mismatches (claims tool was used but wasn't)
            - Context contradictions
            - Overconfidence without evidence
            
            Returns detailed analysis with scores and suggestions.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "The AI response to evaluate"
                    },
                    "context": {
                        "type": "string",
                        "description": "The conversation context (previous messages)"
                    },
                    "tools_available": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tools available to the AI"
                    },
                    "tools_used": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tools actually called"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model that generated the response (e.g., 'claude-sonnet-4')"
                    },
                    "use_accumulated_context": {
                        "type": "boolean",
                        "description": "If true (default), uses accumulated session context in addition to provided context",
                        "default": True
                    }
                },
                "required": ["response"]
            }
        ),
        Tool(
            name="check_hallucinations",
            description="""Quick hallucination check on a response.
            
            Focused check for:
            - Unfounded factual claims
            - Made-up statistics or data
            - Fabricated tool results
            - Invented APIs or functions
            
            Returns a simple list of suspected hallucinations with confidence scores.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "The AI response to check"
                    },
                    "strict_mode": {
                        "type": "boolean",
                        "description": "If true, flags even low-confidence hallucinations",
                        "default": False
                    }
                },
                "required": ["response"]
            }
        ),
        Tool(
            name="verify_tool_consistency",
            description="""Verify that tool usage matches what the response claims.
            
            Checks if the AI response makes claims about tool results
            without actually calling those tools.
            
            Example issues caught:
            - "I checked the database and found..." but no DB query was made
            - "The API returned 200 OK" but no API call occurred
            - "Based on the calculation..." but calculator wasn't used""",
            inputSchema={
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "The AI response"
                    },
                    "tools_available": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tools available"
                    },
                    "tools_used": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tools actually called"
                    }
                },
                "required": ["response", "tools_available", "tools_used"]
            }
        ),
        Tool(
            name="compare_model_responses",
            description="""Compare the same prompt across different models.
            
            Helps you choose the best model by comparing:
            - Hallucination rates
            - Consistency scores
            - Response quality
            - Safety/alignment
            
            Useful during development to decide which model to use.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "responses": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "model": {"type": "string"},
                                "response": {"type": "string"}
                            }
                        },
                        "description": "List of responses from different models"
                    },
                    "context": {
                        "type": "string",
                        "description": "The prompt/context used"
                    }
                },
                "required": ["responses"]
            }
        ),
        Tool(
            name="get_session_report",
            description="""Generate a quality report for the current development session.
            
            Shows:
            - Total responses evaluated
            - Hallucination rate
            - Common issues found
            - Model performance comparison (if multiple models used)
            - Recommendations for improvement""",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID (uses current session if not provided)"
                    },
                    "detailed": {
                        "type": "boolean",
                        "description": "Include detailed breakdown of each issue",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="start_evaluation_session",
            description="""Start a new evaluation session for tracking.

            Creates a new session to track all evaluations.
            Useful for organizing development testing sessions.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Session name (e.g., 'testing-new-prompt')"
                    },
                    "description": {
                        "type": "string",
                        "description": "What you're testing in this session"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_context_stats",
            description="""Get statistics about the accumulated session context.

            Shows:
            - Number of interactions in history
            - Size of compacted history
            - Context version (compaction count)
            - Total evaluations in session

            Useful for monitoring context growth and compaction.""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="clear_context",
            description="""Clear all accumulated context while keeping session active.

            Use this to start fresh within a session without losing evaluation records.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to confirm clearing context",
                        "default": False
                    }
                },
                "required": []
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""

    try:
        if name == "evaluate_response":
            result = await evaluate_response_handler(arguments)
        elif name == "check_hallucinations":
            result = await check_hallucinations_handler(arguments)
        elif name == "verify_tool_consistency":
            result = await verify_tool_consistency_handler(arguments)
        elif name == "compare_model_responses":
            result = await compare_models_handler(arguments)
        elif name == "get_session_report":
            result = await session_report_handler(arguments)
        elif name == "start_evaluation_session":
            result = await start_session_handler(arguments)
        elif name == "get_context_stats":
            result = get_context_stats_handler(arguments)
        elif name == "clear_context":
            result = clear_context_handler(arguments)
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": str(e),
                "tool": name
            }, indent=2)
        )]


async def evaluate_response_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Main evaluation handler using Inspect AI + Petri patterns"""

    response = args["response"]
    provided_context = args.get("context", "")
    tools_available = args.get("tools_available", [])
    tools_used = args.get("tools_used", [])
    model = args.get("model", "unknown")
    use_accumulated = args.get("use_accumulated_context", True)  # Default to using accumulated context

    # Build full context: accumulated session context + provided context
    if use_accumulated:
        accumulated = context_manager.get_accumulated_context()
        if accumulated and provided_context:
            full_context = f"{accumulated}\n\n[Current Context]\n{provided_context}"
        elif accumulated:
            full_context = accumulated
        else:
            full_context = provided_context
    else:
        full_context = provided_context

    # Run comprehensive evaluation with full context
    evaluation = await evaluator.evaluate_comprehensive(
        response=response,
        context=full_context,
        tools_available=tools_available,
        tools_used=tools_used
    )

    # Use Petri-style judge for behavioral assessment
    judge_result = await judge.evaluate(
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
        "pass": evaluation["overall_score"] >= 0.7  # Threshold for "good enough"
    }
    
    # Store evaluation
    storage.store(current_session["id"], result)
    current_session["evaluations"].append(result)

    # Add this interaction to context history for future evaluations
    eval_summary = f"Score: {result['overall_score']:.2f}, Risk: {result['risk_level']}"
    if result["warnings"]:
        eval_summary += f", Warnings: {len(result['warnings'])}"
    context_manager.add_interaction(
        context=provided_context,
        response=response,
        evaluation_summary=eval_summary
    )

    # Add context stats to result
    result["context_stats"] = context_manager.get_context_stats()

    return result


async def check_hallucinations_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Quick hallucination check"""
    
    response = args["response"]
    strict_mode = args.get("strict_mode", False)
    
    hallucinations = await evaluator.detect_hallucinations(
        response=response,
        strict=strict_mode
    )
    
    return {
        "hallucinations_found": len(hallucinations),
        "hallucinations": hallucinations,
        "risk_level": "high" if len(hallucinations) > 3 else "medium" if len(hallucinations) > 0 else "low",
        "summary": f"Found {len(hallucinations)} potential hallucination(s)"
    }


async def verify_tool_consistency_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Verify tool usage consistency"""
    
    response = args["response"]
    tools_available = args["tools_available"]
    tools_used = args["tools_used"]
    
    issues = await evaluator.verify_tool_consistency(
        response=response,
        tools_available=tools_available,
        tools_used=tools_used
    )
    
    return {
        "consistent": len(issues) == 0,
        "issues": issues,
        "tools_mentioned_not_used": [i["tool"] for i in issues if i["type"] == "mentioned_not_used"],
        "summary": f"Found {len(issues)} tool consistency issue(s)"
    }


async def compare_models_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Compare multiple model responses"""
    
    responses = args["responses"]
    context = args.get("context", "")
    
    results = []
    for item in responses:
        eval_result = await evaluator.evaluate_comprehensive(
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
    """Generate session report"""
    
    session_id = args.get("session_id", current_session["id"])
    detailed = args.get("detailed", False)
    
    evaluations = current_session["evaluations"]
    
    if not evaluations:
        return {
            "session_id": session_id,
            "total_evaluations": 0,
            "message": "No evaluations in this session yet"
        }
    
    # Calculate statistics
    total = len(evaluations)
    total_hallucinations = sum(e["dimensions"]["hallucination"]["count"] for e in evaluations)
    avg_score = sum(e["overall_score"] for e in evaluations) / total
    
    # Count issues by type
    issues_summary = {
        "hallucinations": total_hallucinations,
        "tool_mismatches": sum(len(e["dimensions"]["tool_consistency"]["issues"]) for e in evaluations),
        "contradictions": sum(len(e["dimensions"]["context_consistency"]["contradictions"]) for e in evaluations),
        "overconfident_claims": sum(len(e["dimensions"]["confidence_calibration"]["overconfident_claims"]) for e in evaluations)
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
    """Start new evaluation session"""
    
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    name = args.get("name", session_id)
    description = args.get("description", "")
    
    current_session["id"] = session_id
    current_session["name"] = name
    current_session["description"] = description
    current_session["started_at"] = datetime.now().isoformat()
    current_session["history"] = []
    current_session["evaluations"] = []
    
    return {
        "session_id": session_id,
        "name": name,
        "message": f"Started new evaluation session: {name}"
    }


def get_risk_level(score: float) -> str:
    """Calculate risk level from score"""
    if score >= 0.8:
        return "low"
    elif score >= 0.5:
        return "medium"
    else:
        return "high"


def collect_warnings(evaluation: Dict[str, Any]) -> List[str]:
    """Collect all warnings from evaluation"""
    warnings = []
    
    if evaluation["hallucination_score"] < 0.5:
        warnings.append(f"⚠️ High hallucination risk: {len(evaluation['hallucinations'])} potential hallucination(s)")
    
    if evaluation["tool_consistency_score"] < 0.7:
        warnings.append(f"⚠️ Tool consistency issues: {len(evaluation['tool_mismatches'])} mismatch(es)")
    
    if evaluation["context_consistency_score"] < 0.7:
        warnings.append(f"⚠️ Context contradictions: {len(evaluation['contradictions'])} found")
    
    if len(evaluation["overconfident_claims"]) > 2:
        warnings.append(f"⚠️ Overconfidence: {len(evaluation['overconfident_claims'])} unsupported confident claim(s)")
    
    return warnings


def generate_suggestions(evaluation: Dict[str, Any], judge_result: Dict[str, Any]) -> List[str]:
    """Generate actionable suggestions"""
    suggestions = []
    
    # Hallucination suggestions
    if evaluation["hallucinations"]:
        suggestions.append("💡 Add evidence/sources for claims or use more cautious language")
        suggestions.append("💡 Consider adding tool calls to verify facts before stating them")
    
    # Tool consistency suggestions
    if evaluation["tool_mismatches"]:
        for mismatch in evaluation["tool_mismatches"]:
            suggestions.append(f"💡 Actually call {mismatch['tool']} before claiming results from it")
    
    # Context consistency suggestions
    if evaluation["contradictions"]:
        suggestions.append("💡 Review conversation history to ensure consistency")
    
    # Judge-based suggestions
    if judge_result["score"] < 0.7:
        suggestions.append(f"💡 Judge identified issues: {judge_result['summary']}")
    
    return suggestions


def generate_session_recommendations(evaluations: List[Dict[str, Any]]) -> List[str]:
    """Generate session-level recommendations"""
    recommendations = []

    hallucination_rate = sum(e["dimensions"]["hallucination"]["count"] for e in evaluations) / len(evaluations)

    if hallucination_rate > 2:
        recommendations.append("📊 High hallucination rate detected - consider stricter system prompts or different model")

    tool_issues = sum(len(e["dimensions"]["tool_consistency"]["issues"]) for e in evaluations)
    if tool_issues > len(evaluations) * 0.3:
        recommendations.append("📊 Frequent tool consistency issues - review tool calling logic")

    avg_score = sum(e["overall_score"] for e in evaluations) / len(evaluations)
    if avg_score < 0.6:
        recommendations.append("📊 Overall quality below threshold - consider prompt refinement or model change")

    return recommendations


def get_context_stats_handler(_args: Dict[str, Any]) -> Dict[str, Any]:
    """Get context accumulation statistics"""
    stats = context_manager.get_context_stats()

    # Add additional info
    stats["session_id"] = current_session["id"]
    stats["accumulated_context_preview"] = context_manager.get_accumulated_context()[:500] + "..." \
        if len(context_manager.get_accumulated_context()) > 500 else context_manager.get_accumulated_context()

    return stats


def clear_context_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Clear accumulated context"""
    confirm = args.get("confirm", False)

    if not confirm:
        return {
            "cleared": False,
            "message": "Set confirm=true to clear context"
        }

    context_manager.clear()

    return {
        "cleared": True,
        "message": "Context cleared successfully",
        "context_stats": context_manager.get_context_stats()
    }


async def main():
    """Run the MCP server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
