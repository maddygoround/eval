"""
Change Tracking for AI Session Evaluation.

This module tracks actual changes made during an AI session,
including file modifications (via git diff), tool calls, and
their results. This enables verification of AI claims against
actual behavior.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import subprocess
import os


@dataclass
class ToolCall:
    """Record of a single tool call."""
    tool_name: str
    parameters: Dict[str, Any]
    result: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True
    error: Optional[str] = None


@dataclass
class FileChange:
    """Record of a file change with diff."""
    file_path: str
    change_type: str  # "created", "modified", "deleted", "renamed"
    diff: str = ""
    lines_added: int = 0
    lines_removed: int = 0
    old_path: Optional[str] = None  # For renames


class ChangeTracker:
    """
    Tracks changes made during an AI session.

    Captures:
    - Tool calls and their results
    - File changes via git diff
    - Baseline state for comparison
    """

    def __init__(self, working_dir: str = None):
        """
        Initialize the change tracker.

        Args:
            working_dir: Working directory for git operations.
                        Defaults to current directory.
        """
        self.working_dir = working_dir or os.getcwd()
        self.tool_calls: List[ToolCall] = []
        self.baseline_commit: Optional[str] = None
        self.tracked_files: Dict[str, str] = {}  # file_path -> original content hash
        self._snapshot_baseline()

    def _snapshot_baseline(self) -> None:
        """Capture baseline git state."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.working_dir,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self.baseline_commit = result.stdout.strip()
        except Exception:
            self.baseline_commit = None

    def record_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None
    ) -> ToolCall:
        """
        Record a tool call.

        Args:
            tool_name: Name of the tool called (Read, Write, Edit, Bash, etc.)
            parameters: Parameters passed to the tool
            result: Result/output from the tool
            success: Whether the tool call succeeded
            error: Error message if failed

        Returns:
            The recorded ToolCall
        """
        call = ToolCall(
            tool_name=tool_name,
            parameters=parameters,
            result=result,
            success=success,
            error=error
        )
        self.tool_calls.append(call)
        return call

    def get_file_changes(self, file_paths: List[str] = None) -> List[FileChange]:
        """
        Get file changes since baseline using git diff.

        Args:
            file_paths: Specific files to check. If None, checks all changes.

        Returns:
            List of FileChange objects with diffs
        """
        changes = []

        try:
            # Get list of changed files
            if file_paths:
                # Check specific files
                for file_path in file_paths:
                    change = self._get_single_file_change(file_path)
                    if change:
                        changes.append(change)
            else:
                # Get all changes
                changes = self._get_all_git_changes()
        except Exception:
            # If git fails, fall back to checking modified_files directly
            pass

        return changes

    def _get_single_file_change(self, file_path: str) -> Optional[FileChange]:
        """Get change info for a single file."""
        abs_path = os.path.abspath(file_path)

        # Check if file exists
        if not os.path.exists(abs_path):
            return FileChange(
                file_path=file_path,
                change_type="deleted",
                diff="File deleted"
            )

        # Get diff
        try:
            # Try staged diff first
            result = subprocess.run(
                ["git", "diff", "--cached", "--", file_path],
                cwd=self.working_dir,
                capture_output=True,
                text=True
            )
            diff = result.stdout.strip()

            # If no staged changes, try unstaged
            if not diff:
                result = subprocess.run(
                    ["git", "diff", "--", file_path],
                    cwd=self.working_dir,
                    capture_output=True,
                    text=True
                )
                diff = result.stdout.strip()

            # If still no diff, check if it's a new file
            if not diff:
                result = subprocess.run(
                    ["git", "status", "--porcelain", "--", file_path],
                    cwd=self.working_dir,
                    capture_output=True,
                    text=True
                )
                status = result.stdout.strip()
                if status.startswith("??") or status.startswith("A"):
                    # New file - show full content as diff
                    with open(abs_path, 'r') as f:
                        content = f.read()
                    lines = content.splitlines()
                    diff = f"+++ {file_path}\n" + "\n".join(f"+{line}" for line in lines)
                    return FileChange(
                        file_path=file_path,
                        change_type="created",
                        diff=diff,
                        lines_added=len(lines),
                        lines_removed=0
                    )

            if diff:
                # Parse diff stats
                added = diff.count("\n+") - diff.count("\n+++")
                removed = diff.count("\n-") - diff.count("\n---")

                return FileChange(
                    file_path=file_path,
                    change_type="modified",
                    diff=diff,
                    lines_added=max(0, added),
                    lines_removed=max(0, removed)
                )
        except Exception:
            pass

        return None

    def _get_all_git_changes(self) -> List[FileChange]:
        """Get all git changes in working directory."""
        changes = []

        try:
            # Get status
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.working_dir,
                capture_output=True,
                text=True
            )

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                status = line[:2]
                file_path = line[3:].strip()

                # Handle renames
                if " -> " in file_path:
                    old_path, new_path = file_path.split(" -> ")
                    change = self._get_single_file_change(new_path)
                    if change:
                        change.change_type = "renamed"
                        change.old_path = old_path
                        changes.append(change)
                else:
                    change = self._get_single_file_change(file_path)
                    if change:
                        changes.append(change)
        except Exception:
            pass

        return changes

    def get_tool_calls_summary(self) -> Dict[str, Any]:
        """
        Get summary of tool calls for evaluation context.

        Returns:
            Dictionary with tool call statistics and details
        """
        if not self.tool_calls:
            return {"total_calls": 0, "tools": {}, "calls": []}

        # Group by tool name
        tools_summary = {}
        for call in self.tool_calls:
            if call.tool_name not in tools_summary:
                tools_summary[call.tool_name] = {
                    "count": 0,
                    "success_count": 0,
                    "files_affected": set()
                }

            tools_summary[call.tool_name]["count"] += 1
            if call.success:
                tools_summary[call.tool_name]["success_count"] += 1

            # Track files affected
            file_param = call.parameters.get("file_path") or call.parameters.get("path")
            if file_param:
                tools_summary[call.tool_name]["files_affected"].add(file_param)

        # Convert sets to lists for JSON serialization
        for tool in tools_summary.values():
            tool["files_affected"] = list(tool["files_affected"])

        return {
            "total_calls": len(self.tool_calls),
            "tools": tools_summary,
            "calls": [
                {
                    "tool": c.tool_name,
                    "params": c.parameters,
                    "success": c.success,
                    "timestamp": c.timestamp
                }
                for c in self.tool_calls
            ]
        }

    def build_change_context(self, modified_files: List[str] = None) -> str:
        """
        Build comprehensive change context for evaluation.

        This combines:
        - Git diffs showing actual changes
        - Tool call history
        - File state information

        Args:
            modified_files: List of files to include. If None, uses git status.

        Returns:
            Formatted context string for evaluation
        """
        parts = []

        # Add tool call summary
        tool_summary = self.get_tool_calls_summary()
        if tool_summary["total_calls"] > 0:
            parts.append("[TOOL CALLS MADE]")
            for tool_name, stats in tool_summary["tools"].items():
                parts.append(f"- {tool_name}: {stats['count']} calls ({stats['success_count']} successful)")
                if stats["files_affected"]:
                    parts.append(f"  Files: {', '.join(stats['files_affected'][:5])}")
            parts.append("")

        # Add file changes with diffs
        file_changes = self.get_file_changes(modified_files)
        if file_changes:
            parts.append("[FILE CHANGES (git diff)]")
            for change in file_changes:
                parts.append(f"\nFile: {change.file_path}")
                parts.append(f"Type: {change.change_type}")
                parts.append(f"Lines: +{change.lines_added} -{change.lines_removed}")

                # Include diff (truncated if needed)
                if change.diff:
                    diff_lines = change.diff.splitlines()
                    if len(diff_lines) > 100:
                        # Keep first 60 and last 30 lines
                        truncated_diff = "\n".join(
                            diff_lines[:60] +
                            [f"... [{len(diff_lines) - 90} lines omitted] ..."] +
                            diff_lines[-30:]
                        )
                        parts.append(f"```diff\n{truncated_diff}\n```")
                    else:
                        parts.append(f"```diff\n{change.diff}\n```")
            parts.append("")

        return "\n".join(parts)

    def clear(self) -> None:
        """Clear all tracked data and reset baseline."""
        self.tool_calls = []
        self.tracked_files = {}
        self._snapshot_baseline()

    def get_write_operations(self) -> List[Dict[str, Any]]:
        """Get all Write/Edit tool calls for verification."""
        write_ops = []
        for call in self.tool_calls:
            if call.tool_name in ("Write", "Edit", "NotebookEdit"):
                write_ops.append({
                    "tool": call.tool_name,
                    "file_path": call.parameters.get("file_path"),
                    "timestamp": call.timestamp,
                    "success": call.success
                })
        return write_ops
