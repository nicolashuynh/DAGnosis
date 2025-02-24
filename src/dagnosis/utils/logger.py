"""Utility functions for logging configuration."""

# stdlib
import logging

DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logger(
    name: str = __name__, level: int = DEFAULT_LOG_LEVEL
) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        name: Logger name, defaults to module name
        level: Logging level, defaults to INFO

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:  # Avoid adding handlers multiple times
        handler = logging.StreamHandler()
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger
