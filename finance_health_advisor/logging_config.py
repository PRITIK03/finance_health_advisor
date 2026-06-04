"""
Logging Infrastructure for Finance Health Advisor
"""
import logging
import os
import sys
from config import CONFIG


def setup_logger(name: str = "fha", level: str = None) -> logging.Logger:
    """Setup and return a configured logger."""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    log_level = level or CONFIG.log_level or "INFO"
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = "fha") -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)


# Initialize default logger
logger = setup_logger("fha")