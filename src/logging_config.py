"""
src/logging_config.py
Centralized logging configuration for SAP ERP AI Agent.

Call setup_logging() once at the top of every entry point:
  - src/main.py
  - test_*.py scripts

Log files are written to  <project_root>/logs/app.log
with automatic rotation (max_bytes / backup_count defined in configs.yaml).

All logging settings are read from configs.yaml via get_config().
Override at call-site only for testing:
    setup_logging(log_level="WARNING")
"""

import logging
import logging.config
from pathlib import Path

from src.config import get_config


def setup_logging(
    log_level: str | None = None,
    log_dir: str | None = None,
) -> None:
    """
    Configure logging for the entire application.

    Parameters
    ----------
    log_level : str, optional
        Override log level. Falls back to configs.yaml → logging.level.
    log_dir : str, optional
        Override log directory. Falls back to configs.yaml → logging.dir.
    """
    cfg = get_config().logging

    level_str = log_level or cfg.level
    console_level_str = cfg.console_level
    level = getattr(logging, level_str.upper(), logging.DEBUG)

    # Resolve log directory relative to *project root* (two levels up from this file)
    project_root = Path(__file__).resolve().parent.parent
    resolved_log_dir = Path(log_dir or cfg.dir)
    if not resolved_log_dir.is_absolute():
        resolved_log_dir = project_root / resolved_log_dir
    resolved_log_dir.mkdir(parents=True, exist_ok=True)

    log_file = resolved_log_dir / "app.log"

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            # Concise format for the console
            "console": {
                "format": "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                "datefmt": "%H:%M:%S",
            },
            # Detailed format for the log file
            "file": {
                "format": (
                    "%(asctime)s  %(levelname)-8s  %(name)s"
                    "  [%(filename)s:%(lineno)d]  %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": console_level_str,
                "formatter": "console",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": level_str.upper(),
                "formatter": "file",
                "filename": str(log_file),
                "maxBytes": cfg.max_bytes,
                "backupCount": cfg.backup_count,
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": level,
            "handlers": ["console", "file"],
        },
        # Silence overly verbose third-party loggers
        "loggers": {
            "httpx":          {"level": "WARNING", "propagate": True},
            "httpcore":       {"level": "WARNING", "propagate": True},
            "langchain":      {"level": "WARNING", "propagate": True},
            "langchain_core": {"level": "WARNING", "propagate": True},
            "ollama":         {"level": "WARNING", "propagate": True},
        },
    }

    logging.config.dictConfig(LOGGING_CONFIG)
    logging.getLogger(__name__).info(
        "Logging initialised — level=%s  file=%s", level_str.upper(), log_file
    )
