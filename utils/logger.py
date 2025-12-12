"""
Logger Setup
Configure logging for training and evaluation.
"""
import logging
import sys
from typing import Optional
import os


def setup_logger(
    name: str = "nmt",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TensorBoardLogger:
    """
    TensorBoard logging wrapper.
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            logging.warning("TensorBoard not available")
            self.writer = None
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars."""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram of values."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text."""
        if self.enabled:
            self.writer.add_text(tag, text, step)
    
    def close(self):
        """Close the writer."""
        if self.enabled:
            self.writer.close()


if __name__ == "__main__":
    logger = setup_logger("test", log_file="/tmp/test.log")
    logger.info("Test message")
    logger.warning("Test warning")
