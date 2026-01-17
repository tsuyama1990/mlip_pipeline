import logging
from pathlib import Path

import pytest

from mlip_autopipec.core.logging import setup_logging


def test_setup_logging(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test that logging is set up correctly."""
    log_file = tmp_path / "test.log"
    setup_logging(log_file)

    logger = logging.getLogger("test_logger")
    logger.info("Test log message")

    # Verify file output
    assert log_file.exists()
    content = log_file.read_text()
    assert "Test log message" in content
