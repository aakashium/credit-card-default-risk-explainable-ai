import logging
import os
from pathlib import Path

def setup_logger(name: str, log_file: str = "logs/app.log", level: int = logging.INFO):
    """
    Sets up a logger with both console and file handlers.

    Parameters
    ----------
    name : str
        Name of the logger (usually __name__ of the caller module).
    log_file : str, optional
        Path to log file (default = logs/app.log).
    level : int, optional
        Logging level (default = logging.INFO).

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
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Format
        formatter = logging.Formatter(
            "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
