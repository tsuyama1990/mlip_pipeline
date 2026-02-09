import logging
from pathlib import Path

import pytest

from mlip_autopipec.core.logger import setup_logging


def test_setup_logging_console(capsys: pytest.CaptureFixture[str]) -> None:
    """Test setting up console logging."""
    logger = setup_logging(name="test_console")
    # Reset handlers for clean test environment or check existing
    # Logger instance is global, so it persists. We should use unique names.

    assert logger.name == "test_console"
    assert logger.level == logging.INFO
    # Depending on previous tests, handlers might be there if names collide.
    # But here we used unique name "test_console".
    assert len(logger.handlers) >= 1

    # Check if console handler is present
    has_console = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert has_console

    logger.info("Test message")
    # Capturing logging output via capsys is tricky because logging might write to stderr or stdout
    # and pytest captures it.
    captured = capsys.readouterr()
    # By default StreamHandler writes to sys.stderr.
    assert "Test message" in captured.err or "Test message" in captured.out


def test_setup_logging_file(tmp_path: Path) -> None:
    """Test setting up file logging."""
    log_file = tmp_path / "test.log"
    logger = setup_logging(name="test_file", log_file=log_file)

    assert log_file.exists()

    logger.info("File message")

    # We need to flush handlers to ensure write
    for h in logger.handlers:
        h.flush()
        h.close()  # Close to release file handle

    content = log_file.read_text()
    assert "File message" in content


def test_duplicate_handlers() -> None:
    """Test that calling setup_logging twice doesn't duplicate handlers."""
    logger1 = setup_logging(name="test_duplicate")
    initial_handlers = len(logger1.handlers)

    logger2 = setup_logging(name="test_duplicate")
    assert len(logger2.handlers) == initial_handlers
    assert logger1 is logger2
