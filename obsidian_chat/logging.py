"""Logging configuration for obsidian-chat."""

import sys
from pathlib import Path

from loguru import logger

# Remove default handler
logger.remove()

# Default format for console
CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# Simpler format for file logging
FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
    "{name}:{function}:{line} | {message}"
)


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    json_logs: bool = False,
) -> None:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path to write logs to.
        json_logs: If True, output logs as JSON (useful for log aggregation).
    """
    # Console handler
    logger.add(
        sys.stderr,
        format=CONSOLE_FORMAT,
        level=level,
        colorize=True,
    )

    # File handler if specified
    if log_file:
        logger.add(
            log_file,
            format=FILE_FORMAT if not json_logs else None,
            level=level,
            rotation="10 MB",
            retention="7 days",
            serialize=json_logs,
        )


def get_logger(name: str = "obsidian_chat"):
    """Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__ of the module).

    Returns:
        Configured logger instance.
    """
    return logger.bind(name=name)


# Default setup for module imports
setup_logging()
