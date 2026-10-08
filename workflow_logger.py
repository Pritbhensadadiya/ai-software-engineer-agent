import logging


LOGGER_NAME = "ai_software_engineer"


def get_workflow_logger() -> logging.Logger:
    """Return the shared workflow logger."""
    logger = logging.getLogger(LOGGER_NAME)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    return logger
