"""Central logging configuration for the classifier pipeline.

This module handles FILE LOGGING ONLY - for console output, use utils.console.

Usage:
    from utils.logging import get_logger, init_logging
    init_logging("main")  # once at app start (e.g., in main.py)
    logger = get_logger(__name__)
    logger.debug("Detailed debug info")  # Goes to file only
    logger.info("Important event")       # Goes to file only

Features:
    * File only: detailed with timestamp, module, function, line number
    * New log file per run (./logs/{name}_YYYYmmdd_HHMMSS.log)
    * DEBUG level captures everything for forensic analysis
    * Structured format that's grep-able and parseable

For user-facing terminal output, use:
    from utils.console import console
    console.start("Pipeline started")
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path

_INITIALIZED = False
LOG_DIR = Path("logs")
# Default log file, updated by init_logging
LOG_FILE = LOG_DIR / "pipeline.log"
DEFAULT_FILE_LEVEL = logging.DEBUG


def init_logging(name: str = "pipeline", file_level: int = DEFAULT_FILE_LEVEL) -> None:
    """Initialize file logging once. Safe to call multiple times.
    
    Args:
        name: Base name for the log file (e.g. 'main', 'test'). 
              A timestamp will be appended: logs/{name}_{date}.log
        file_level: Minimum level for file logging (default: DEBUG)
    
    Note:
        Console output is handled separately by utils.console module.
        This function only sets up file logging.
    """
    global _INITIALIZED, LOG_FILE
    if _INITIALIZED:
        return
    LOG_DIR.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Capture all at root level

    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = LOG_DIR / f"{name}_{timestamp}.log"

    # File handler: detailed format for debugging
    # Format: timestamp | level | module:function:line | message
    file_fmt = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    
    # Using FileHandler instead of RotatingFileHandler since we create a new file each run
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter(file_fmt, datefmt="%Y-%m-%d %H:%M:%S")
    )
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name.
    
    Args:
        name: Usually __name__ from the calling module
        
    Returns:
        Logger instance configured to write to file only
    """
    return logging.getLogger(name)


__all__ = ["init_logging", "get_logger", "LOG_FILE"]
