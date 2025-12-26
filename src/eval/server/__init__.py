"""
MCP Server module for AI Evaluator Framework.

Import from submodules directly:
    from eval.server.app import create_server, run_server
    from eval.server.handlers import evaluate_response_handler
"""

# Lazy imports to avoid circular import warnings when running as __main__
def __getattr__(name):
    if name in ("create_server", "run_server"):
        from .app import create_server, run_server
        return {"create_server": create_server, "run_server": run_server}[name]
    if name in (
        "evaluate_response_handler",
        "check_hallucinations_handler",
        "verify_tool_consistency_handler",
        "compare_models_handler",
        "session_report_handler",
        "start_session_handler",
        "get_context_stats_handler",
        "clear_context_handler",
    ):
        from . import handlers
        return getattr(handlers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "create_server",
    "run_server",
    "evaluate_response_handler",
    "check_hallucinations_handler",
    "verify_tool_consistency_handler",
    "compare_models_handler",
    "session_report_handler",
    "start_session_handler",
    "get_context_stats_handler",
    "clear_context_handler",
]
