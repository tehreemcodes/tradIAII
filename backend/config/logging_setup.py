"""
Logging Setup
==============
Windows-safe UTF-8 logging that prevents CP1252 encoding crashes
when logging messages contain non-ASCII characters (arrows, symbols).

Usage in every script:
    from backend.config.logging_setup import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
"""
import io
import sys
import logging
from backend.config.settings import LOG_FILE, LOG_FORMAT, LOG_LEVEL


def setup_logging(level: str = LOG_LEVEL) -> None:
    """
    Configure root logger with UTF-8 safe stream and file handlers.
    Safe on Windows (CP1252), Linux, and macOS.
    Idempotent — safe to call multiple times.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # UTF-8 wrapped stdout — prevents CP1252 crash on Windows
    try:
        utf8_stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
        stream_handler = logging.StreamHandler(utf8_stdout)
    except AttributeError:
        # Fallback for environments without .buffer
        stream_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(LOG_FORMAT)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()

    root.setLevel(numeric_level)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
