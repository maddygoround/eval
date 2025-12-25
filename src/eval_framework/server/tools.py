"""
MCP Tool Definitions for AI Evaluator.

This module defines the tools exposed by the MCP server.
"""

from mcp.types import Tool


def get_tools() -> list[Tool]:
    """Get all available MCP tools."""
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
                    "modified_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of absolute paths to files modified by the agent"
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
