"""
Evaluation Storage
Tracks and stores evaluation results across sessions
"""

import json
import sqlite3
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path


class EvaluationStorage:
    """
    Store and retrieve evaluation results
    Uses SQLite for persistence
    """
    
    def __init__(self, db_path: str = "./evaluations.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        
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
        
        # Issues table (for detailed issue tracking)
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
        """Store an evaluation result"""
        
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
            
            # Store issues
            warnings = evaluation.get("warnings", [])
            for warning in warnings:
                cursor.execute("""
                    INSERT INTO issues (
                        evaluation_id,
                        issue_type,
                        severity,
                        description
                    ) VALUES (?, ?, ?, ?)
                """, (
                    evaluation_id,
                    "warning",
                    "medium",
                    warning
                ))
            
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
        """Get statistics for a session"""
        
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
    
    def get_recent_evaluations(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent evaluations for a session"""
        
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
    
    def create_session(self, session_id: str, name: str = "", description: str = ""):
        """Create a new session"""
        
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
    
    def end_session(self, session_id: str):
        """Mark session as ended"""
        
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
    
    def export_session(self, session_id: str, output_path: str):
        """Export session data to JSON"""
        
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
