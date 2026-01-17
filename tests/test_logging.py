import pytest
import logging
import os
from mlip_autopipec.core.logging import setup_logging

def test_setup_logging_creates_file(tmp_path):
    log_file = tmp_path / "system.log"
    setup_logging(log_file)

    logger = logging.getLogger("mlip_autopipec")
    logger.info("Test message")

    assert log_file.exists()
    content = log_file.read_text()
    assert "Test message" in content

def test_setup_logging_formatting(tmp_path):
    log_file = tmp_path / "system.log"
    setup_logging(log_file)

    logger = logging.getLogger("mlip_autopipec")
    logger.info("Formatted message")

    content = log_file.read_text()
    # Check if timestamp exists (roughly) or level name
    assert "INFO" in content
    assert "Formatted message" in content
