# unified_concept_editing/logger.py

import logging

def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Setup a logger for the module.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): File to log to. Defaults to None.
        level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers = [handler, file_handler]
    else:
        handlers = [handler]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        for h in handlers:
            logger.addHandler(h)

    return logger
