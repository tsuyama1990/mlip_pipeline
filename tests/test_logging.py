import pytest
import logging
from pathlib import Path
from mlip_autopipec.core.logging import setup_logging

def test_setup_logging(tmp_path):
    log_path = tmp_path / "test.log"
    setup_logging(log_path, level="DEBUG")

    assert log_path.exists()

    # We cannot use caplog here easily because setup_logging clears handlers.
    # We verify via file output.

    logging.debug("This is a debug message")
    logging.info("This is an info message")

    # Force flush not needed for file handler usually, but let's be safe if buffering
    # logging.shutdown() # Don't shutdown, might break other tests

    # Check if log file contains messages
    with open(log_path, 'r') as f:
        content = f.read()
        assert "This is a debug message" in content
        assert "This is an info message" in content

def test_logging_level(tmp_path):
    log_path = tmp_path / "test_level.log"
    setup_logging(log_path, level="WARNING")

    logging.info("Should not see this")
    logging.warning("Should see this")

    with open(log_path, 'r') as f:
        content = f.read()
        assert "Should not see this" not in content
        assert "Should see this" in content
