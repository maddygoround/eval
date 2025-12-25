"""
MCP Server Application.

This module creates and runs the MCP server for AI evaluation.
"""

import asyncio
import json
from typing import Any

import nest_asyncio

# Apply nest_asyncio BEFORE any event loop starts
# This is critical for running Inspect AI's async eval within MCP server's event loop
nest_asyncio.apply()

from mcp.server import Server
from mcp.types import TextContent
import mcp.server.stdio

from .tools import get_tools
from .handlers import (
    evaluate_response_handler,
    check_hallucinations_handler,
    verify_tool_consistency_handler,
    compare_models_handler,
    session_report_handler,
    start_session_handler,
    get_context_stats_handler,
    clear_context_handler,
)


def create_server() -> Server:
    """
    Create and configure the MCP server.

    Returns:
        Configured Server instance.
    """
    app = Server("ai-evaluator")

    @app.list_tools()
    async def list_tools():
        """List available evaluation tools."""
        return get_tools()

    @app.call_tool()
    async def call_tool(name: str, arguments: Any) -> list[TextContent]:
        """Handle tool calls."""
        try:
            # Route to appropriate handler
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

    return app


async def run_server() -> None:
    """Run the MCP server."""
    app = create_server()

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


def main() -> None:
    """Main entry point for the server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
