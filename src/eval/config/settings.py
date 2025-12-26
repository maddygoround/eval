"""
Configuration settings for the AI Evaluator Framework.
"""

from dataclasses import dataclass, field
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ContextConfig:
    """Configuration for context management and compaction."""
    max_history_items: int = 20
    max_context_chars: int = 15000
    compaction_target_chars: int = 5000
    keep_recent_items: int = 3


@dataclass
class EvaluatorConfig:
    """Configuration for the evaluator."""
    judge_model: str = "anthropic/claude-sonnet-4-5-20250929"
    compaction_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1000
    temperature: float = 0.3
    log_dir: str = os.path.expanduser("~/.eval/logs")
    pass_threshold: float = 0.7


@dataclass
class DatabaseConfig:
    """Configuration for database storage."""
    db_path: str = os.path.expanduser("~/.eval/evaluations.db")


@dataclass
class Settings:
    """Main settings container."""
    context: ContextConfig = field(default_factory=ContextConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    # API keys from environment
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        # Define defaults with expansion
        default_log_dir = os.path.expanduser("~/.eval/logs")
        default_db_path = os.path.expanduser("~/.eval/evaluations.db")

        # Get from env or use expanded defaults
        log_dir = os.path.expanduser(os.getenv("LOG_DIR", default_log_dir))
        db_path = os.path.expanduser(os.getenv("DB_PATH", default_db_path))

        # Ensure directories exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        return cls(
            context=ContextConfig(
                max_history_items=int(os.getenv("MAX_HISTORY_ITEMS", 20)),
                max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", 15000)),
                compaction_target_chars=int(os.getenv("COMPACTION_TARGET_CHARS", 5000)),
                keep_recent_items=int(os.getenv("KEEP_RECENT_ITEMS", 3)),
            ),
            evaluator=EvaluatorConfig(
                judge_model=os.getenv("JUDGE_MODEL", "anthropic/claude-sonnet-4-5-20250929"),
                pass_threshold=float(os.getenv("PASS_THRESHOLD", 0.7)),
                log_dir=log_dir,
            ),
            database=DatabaseConfig(
                db_path=db_path,
            ),
        )


# Global settings instance
settings = Settings.from_env()
