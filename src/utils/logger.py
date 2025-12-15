"""
logger.py
---------
Enhanced logging configuration with rotation and structured logging support.
"""

import logging
import os
import json
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def setup_logger(
    name: str,
    log_file: str = "logs/app.log",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    use_json: bool = False,
    console_output: bool = True
) -> logging.Logger:
    """
    Sets up a logger with both console and file handlers with rotation.

    Parameters
    ----------
    name : str
        Name of the logger (usually __name__ of the caller module).
    log_file : str, optional
        Path to log file (default = logs/app.log).
    level : int, optional
        Logging level (default = logging.INFO).
    max_bytes : int, optional
        Maximum size of log file before rotation (default = 10 MB).
    backup_count : int, optional
        Number of backup files to keep (default = 5).
    use_json : bool, optional
        Whether to use JSON formatting (default = False).
    console_output : bool, optional
        Whether to output logs to console (default = True).

    Returns
    -------
    logger : logging.Logger
        Configured logger instance.
    """
    # Ensure log directory exists
    log_path = Path(log_file).parent
    os.makedirs(log_path, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers when re-imported
    if logger.handlers:
        return logger

    # Choose formatter
    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s — %(name)s — %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        mode='a',
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    use_config: bool = True
) -> logging.Logger:
    """
    Get a logger instance with optional configuration from config.py.
    
    Parameters
    ----------
    name : str
        Name of the logger.
    log_file : str, optional
        Path to log file. If None and use_config is True, uses config.
    use_config : bool, optional
        Whether to use centralized config (default = True).
        
    Returns
    -------
    logger : logging.Logger
        Configured logger instance.
    """
    if use_config:
        try:
            from src.utils.config import get_logging_config, get_path_config
            
            log_config = get_logging_config()
            path_config = get_path_config()
            
            # Determine log file path
            if log_file is None:
                log_file = str(path_config.LOGS_DIR / "app.log")
            
            # Get log level from string
            level = getattr(logging, log_config.LOG_LEVEL.upper(), logging.INFO)
            
            return setup_logger(
                name=name,
                log_file=log_file,
                level=level,
                max_bytes=log_config.MAX_BYTES,
                backup_count=log_config.BACKUP_COUNT,
                use_json=log_config.USE_JSON_LOGGING
            )
        except ImportError:
            # Fallback if config is not available
            pass
    
    # Default setup without config
    return setup_logger(name=name, log_file=log_file or "logs/app.log")
