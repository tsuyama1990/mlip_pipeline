import logging
from pathlib import Path

from mlip_autopipec.core.logging import setup_logging


def test_setup_logging_defaults(tmp_path: Path) -> None:
    """Test that logging is set up correctly with defaults."""
    log_file = tmp_path / "test.log"
    setup_logging(log_file)

    logger = logging.getLogger("test_logger")
    logger.info("Test log message")
    logger.debug("Debug message should not appear by default")

    # Verify file output
    assert log_file.exists()
    content = log_file.read_text()
    assert "Test log message" in content
    # Default level is INFO, so debug should not be there (unless captured by root logger differently)
    # The setup_logging sets basicConfig level=INFO.
    # However, existing loggers might need to be reset in tests.
    # basicConfig does nothing if root logger is already configured.
    # setup_logging uses force=True, so it should work.


def test_setup_logging_with_level(tmp_path: Path) -> None:
    """Test that logging level can be customized."""
    log_file = tmp_path / "debug.log"
    setup_logging(log_file, level="DEBUG")

    logger = logging.getLogger("debug_test_logger")
    logger.debug("This is a debug message")

    assert log_file.exists()
    content = log_file.read_text()
    assert "This is a debug message" in content
