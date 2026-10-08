import logging

from workflow_logger import get_workflow_logger


def test_workflow_logger_returns_named_logger():
    logger = get_workflow_logger()

    assert logger.name == "ai_software_engineer"
    assert logger.level == logging.INFO


def test_workflow_logger_reuses_logger():
    first = get_workflow_logger()
    second = get_workflow_logger()

    assert first is second
    assert len(first.handlers) == 1
