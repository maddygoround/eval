"""
Evaluation Storage Module.

Provides persistent storage for evaluation results using SQLite.
"""

import json
import sqlite3
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from .config.settings import settings


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
        # Ensure db directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
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
                petri_score REAL,
                risk_level TEXT,
                passed BOOLEAN,
                response_text TEXT,
                context_text TEXT,
                evaluation_data TEXT
            )
        """)

        # Schema Migration
        self._migrate_db(cursor)

        # Issues table (normalized for queries)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id INTEGER,
                type TEXT,
                description TEXT,
                severity TEXT,
                context TEXT,
                FOREIGN KEY(evaluation_id) REFERENCES evaluations(id)
            )
        """)

        # Petri Dimensions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS petri_dimensions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id INTEGER,
                dimension TEXT,
                score REAL,
                evidence TEXT,
                FOREIGN KEY(evaluation_id) REFERENCES evaluations(id)
            )
        """)

    def _migrate_db(self, cursor: sqlite3.Cursor) -> None:
        """Perform schema migrations if needed."""
        # Get existing columns
        cursor.execute("PRAGMA table_info(evaluations)")
        columns = {col[1] for col in cursor.fetchall()}

        # Add petri_score if missing
        if "petri_score" not in columns:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN petri_score REAL DEFAULT 0.0")

        # Add response_text if missing
        if "response_text" not in columns:
             cursor.execute("ALTER TABLE evaluations ADD COLUMN response_text TEXT DEFAULT ''")

        # Add context_text if missing
        if "context_text" not in columns:
             cursor.execute("ALTER TABLE evaluations ADD COLUMN context_text TEXT DEFAULT ''")

        # Add evaluation_data if missing
        if "evaluation_data" not in columns:
             cursor.execute("ALTER TABLE evaluations ADD COLUMN evaluation_data TEXT DEFAULT ''")



    def save_evaluation(self, result: Dict[str, Any], session_id: str) -> int:
        """
        Save an evaluation result.

        Args:
            result: The evaluation result dictionary (from EvaluationResult.to_dict()).
            session_id: The current session ID.

        Returns:
            The ID of the inserted evaluation record.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Insert main record
            cursor.execute("""
                INSERT INTO evaluations (
                    session_id, timestamp, model, overall_score,
                    hallucination_score, tool_consistency_score,
                    context_consistency_score, confidence_score,
                    petri_score, risk_level, passed, 
                    response_text, context_text, evaluation_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                result.get("timestamp", datetime.now().isoformat()),
                result.get("model", "unknown"),
                result.get("overall_score", 0.0),
                result.get("hallucination_score", 0.0),
                result.get("tool_consistency_score", 0.0),
                result.get("context_consistency_score", 0.0),
                result.get("confidence_score", 0.0),
                result.get("petri_score", 0.0),
                result.get("risk_level", "unknown"),
                result.get("pass", False),
                result.get("response", ""),
                result.get("context", ""),
                json.dumps(result)
            ))
            
            eval_id = cursor.lastrowid

            # Insert Hallucinations
            for h in result.get("hallucinations", []):
                cursor.execute("""
                    INSERT INTO issues (evaluation_id, type, description, severity, context)
                    VALUES (?, ?, ?, ?, ?)
                """, (eval_id, "hallucination", h.get("claim"), "high", h.get("reason")))

            # Insert Tool Issues
            for t_issue in result.get("tool_mismatches", []):
                cursor.execute("""
                    INSERT INTO issues (evaluation_id, type, description, severity, context)
                    VALUES (?, ?, ?, ?, ?)
                """, (eval_id, "tool_mismatch", t_issue.get("details"), "medium", ""))

            # Insert Petri Dimensions
            petri_data = result.get("petri_eval", {})
            for dim in petri_data.get("dimensions", []):
                cursor.execute("""
                    INSERT INTO petri_dimensions (evaluation_id, dimension, score, evidence)
                    VALUES (?, ?, ?, ?)
                """, (eval_id, dim.get("name"), dim.get("score"), dim.get("evidence")))

            # Update Session Stats
            self._update_session_stats(cursor, session_id)

            conn.commit()
            return eval_id

        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _update_session_stats(self, cursor: sqlite3.Cursor, session_id: str) -> None:
        """Update aggregate stats for the session."""
        cursor.execute("""
            SELECT COUNT(*), AVG(overall_score)
            FROM evaluations
            WHERE session_id = ?
        """, (session_id,))
        
        count, avg = cursor.fetchone()
        
        cursor.execute("""
            UPDATE sessions
            SET total_evaluations = ?, average_score = ?
            WHERE session_id = ?
        """, (count, avg if avg else 0.0, session_id))

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session.

        Args:
            session_id: The session ID.

        Returns:
            Dictionary with session stats.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else {}
        finally:
            conn.close()

    def get_recent_evaluations(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent evaluations for a session.

        Args:
            session_id: The session ID.
            limit: Number of records to return.

        Returns:
            List of evaluation dictionaries.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT * FROM evaluations
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            """, (session_id, limit))
            
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
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
