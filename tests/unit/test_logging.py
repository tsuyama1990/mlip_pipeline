import logging
from pathlib import Path

import pytest

from mlip_autopipec.logging_config import setup_logging


def test_setup_logging_defaults() -> None:
    setup_logging()
    logger = logging.getLogger()
    assert logger.level == logging.INFO


def test_setup_logging_custom_level() -> None:
    setup_logging(log_level="DEBUG")
    logger = logging.getLogger()
    assert logger.level == logging.DEBUG


def test_setup_logging_invalid_level() -> None:
    with pytest.raises(ValueError, match="Invalid log level"):
        setup_logging(log_level="INVALID")


def test_setup_logging_file(temp_dir: Path) -> None:
    log_file = temp_dir / "test.log"
    setup_logging(log_file=log_file)

    logger = logging.getLogger("test_file_logger")
    logger.info("Test message")

    # Force flush
    for handler in logging.root.handlers:
        handler.flush()

    assert log_file.exists()
    content = log_file.read_text()
    assert "Test message" in content
