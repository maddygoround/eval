"""
Evaluation Storage Module.

Provides persistent storage for evaluation results using SQLite.
"""

import json
import sqlite3
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from ..config.settings import settings


class EvaluationStorage:
    """
    Store and retrieve evaluation results.

    Uses SQLite for persistence with support for:
    - Session management
    - Evaluation tracking
    - Issue logging
    - Statistics and reporting
    """

    def __init__(self, db_path: str = None):
        """
        Initialize the storage.

        Args:
            db_path: Path to the SQLite database file.
                    Defaults to settings.database.db_path.
        """
        self.db_path = db_path or settings.database.db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                started_at TEXT,
                ended_at TEXT,
                total_evaluations INTEGER DEFAULT 0,
                average_score REAL DEFAULT 0.0
            )
        """)

        # Evaluations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                model TEXT,
                overall_score REAL,
                hallucination_score REAL,
                tool_consistency_score REAL,
                context_consistency_score REAL,
                confidence_score REAL,
                risk_level TEXT,
                passed BOOLEAN,
                response_text TEXT,
                context_text TEXT,
                evaluation_data TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        # Issues table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id INTEGER,
                issue_type TEXT,
                severity TEXT,
                description TEXT,
                suggestion TEXT,
                FOREIGN KEY (evaluation_id) REFERENCES evaluations(id)
            )
        """)

        conn.commit()
        conn.close()

    def store(self, session_id: str, evaluation: Dict[str, Any]) -> int:
        """
        Store an evaluation result.

        Args:
            session_id: The session this evaluation belongs to.
            evaluation: The evaluation result dictionary.

        Returns:
            The ID of the stored evaluation.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Store main evaluation
            cursor.execute("""
                INSERT INTO evaluations (
                    session_id,
                    timestamp,
                    model,
                    overall_score,
                    hallucination_score,
                    tool_consistency_score,
                    context_consistency_score,
                    confidence_score,
                    risk_level,
                    passed,
                    evaluation_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                evaluation.get("timestamp", datetime.now().isoformat()),
                evaluation.get("model", "unknown"),
                evaluation.get("overall_score", 0.0),
                evaluation.get("dimensions", {}).get("hallucination", {}).get("score", 0.0),
                evaluation.get("dimensions", {}).get("tool_consistency", {}).get("score", 0.0),
                evaluation.get("dimensions", {}).get("context_consistency", {}).get("score", 0.0),
                evaluation.get("dimensions", {}).get("confidence_calibration", {}).get("score", 0.0),
                evaluation.get("risk_level", "unknown"),
                evaluation.get("pass", False),
                json.dumps(evaluation)
            ))

            evaluation_id = cursor.lastrowid

            # Store issues/warnings
            warnings = evaluation.get("warnings", [])
            for warning in warnings:
                cursor.execute("""
                    INSERT INTO issues (
                        evaluation_id,
                        issue_type,
                        severity,
                        description
                    ) VALUES (?, ?, ?, ?)
                """, (evaluation_id, "warning", "medium", warning))

            # Update session stats
            cursor.execute("""
                UPDATE sessions
                SET total_evaluations = total_evaluations + 1,
                    average_score = (
                        SELECT AVG(overall_score)
                        FROM evaluations
                        WHERE session_id = ?
                    )
                WHERE session_id = ?
            """, (session_id, session_id))

            conn.commit()
            return evaluation_id

        finally:
            conn.close()

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session.

        Args:
            session_id: The session to get stats for.

        Returns:
            Dictionary with session statistics.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Session info
            cursor.execute("""
                SELECT name, description, started_at, total_evaluations, average_score
                FROM sessions
                WHERE session_id = ?
            """, (session_id,))

            row = cursor.fetchone()
            if not row:
                return {}

            # Issue counts
            cursor.execute("""
                SELECT issue_type, COUNT(*)
                FROM issues
                WHERE evaluation_id IN (
                    SELECT id FROM evaluations WHERE session_id = ?
                )
                GROUP BY issue_type
            """, (session_id,))

            issues_by_type = dict(cursor.fetchall())

            # Risk distribution
            cursor.execute("""
                SELECT risk_level, COUNT(*)
                FROM evaluations
                WHERE session_id = ?
                GROUP BY risk_level
            """, (session_id,))

            risk_distribution = dict(cursor.fetchall())

            return {
                "session_id": session_id,
                "name": row[0],
                "description": row[1],
                "started_at": row[2],
                "total_evaluations": row[3],
                "average_score": row[4],
                "issues_by_type": issues_by_type,
                "risk_distribution": risk_distribution
            }

        finally:
            conn.close()

    def get_recent_evaluations(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent evaluations for a session.

        Args:
            session_id: The session to get evaluations for.
            limit: Maximum number of evaluations to return.

        Returns:
            List of evaluation dictionaries.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, timestamp, model, overall_score, risk_level, passed, evaluation_data
                FROM evaluations
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit))

            results = []
            for row in cursor.fetchall():
                eval_data = json.loads(row[6]) if row[6] else {}
                results.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "model": row[2],
                    "score": row[3],
                    "risk_level": row[4],
                    "passed": bool(row[5]),
                    "data": eval_data
                })

            return results

        finally:
            conn.close()

    def create_session(
        self,
        session_id: str,
        name: str = "",
        description: str = ""
    ) -> None:
        """
        Create a new session.

        Args:
            session_id: Unique identifier for the session.
            name: Human-readable name for the session.
            description: Description of what's being tested.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO sessions (session_id, name, description, started_at)
                VALUES (?, ?, ?, ?)
            """, (session_id, name, description, datetime.now().isoformat()))

            conn.commit()

        finally:
            conn.close()

    def end_session(self, session_id: str) -> None:
        """
        Mark session as ended.

        Args:
            session_id: The session to end.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE sessions
                SET ended_at = ?
                WHERE session_id = ?
            """, (datetime.now().isoformat(), session_id))

            conn.commit()

        finally:
            conn.close()

    def export_session(self, session_id: str, output_path: str) -> str:
        """
        Export session data to JSON.

        Args:
            session_id: The session to export.
            output_path: Path for the output file.

        Returns:
            The output path.
        """
        stats = self.get_session_stats(session_id)
        evaluations = self.get_recent_evaluations(session_id, limit=1000)

        export_data = {
            "session": stats,
            "evaluations": evaluations,
            "exported_at": datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        return output_path
