import logging
from pathlib import Path

def setup_logger(log_file: str = "erase_diff_training.log", level: int = logging.INFO) -> logging.Logger:
    """
    Setup a logger for the training process.

    Args:
        log_file (str, optional): Path to the log file. Defaults to "erase_diff_training.log".
        level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger
