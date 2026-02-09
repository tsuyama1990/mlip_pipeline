import logging
import pytest
from pathlib import Path
from mlip_autopipec.core.logger import setup_logging

def test_setup_logging(tmp_path):
    work_dir = tmp_path / "work_dir"
    work_dir.mkdir()

    logger = logging.getLogger()
    # Clear existing handlers to force setup
    original_handlers = logger.handlers[:]
    logger.handlers = []

    try:
        setup_logging(work_dir)

        log_file = work_dir / "orchestrator.log"
        assert log_file.exists()

        assert len(logger.handlers) >= 2

        # Test logging
        logging.info("Test Info Message")
        logging.debug("Test Debug Message")

        # Flush handlers
        for h in logger.handlers:
            h.flush()

        # Read log file
        with open(log_file, "r") as f:
            content = f.read()
            assert "Test Info Message" in content
            assert "Test Debug Message" in content

        # Ensure idempotency
        current_handler_count = len(logger.handlers)
        setup_logging(work_dir)
        assert len(logger.handlers) == current_handler_count

    finally:
        # Restore handlers
        for h in logger.handlers:
            h.close()
        logger.handlers = original_handlers
