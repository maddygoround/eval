"""
Command-line interface for AI Evaluator Framework.

Provides commands for running the server and evaluation utilities.
"""

import argparse
import asyncio
import sys

from .server.app import run_server


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Evaluator Framework - Evaluate AI responses for quality and safety"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server command
    server_parser = subparsers.add_parser(
        "server",
        help="Run the MCP server"
    )
    server_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    # Version command
    subparsers.add_parser("version", help="Show version information")

    args = parser.parse_args()

    if args.command == "server":
        print("Starting AI Evaluator MCP Server...", file=sys.stderr)
        asyncio.run(run_server())

    elif args.command == "version":
        from . import __version__
        print(f"AI Evaluator Framework v{__version__}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
