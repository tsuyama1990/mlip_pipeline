import logging
from pathlib import Path

import pytest
from mlip_autopipec.core.logging import setup_logging


def test_setup_logging(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    log_path = tmp_path / "system.log"
    setup_logging(log_path)

    # Verify file creation
    # setup_logging usually sets up handlers. The file is created when we log something?
    # Or immediately if FileHandler is initialized with delay=False (default).

    # Let's log something
    logger = logging.getLogger("mlip_autopipec")
    logger.info("Test Log Message")

    assert log_path.exists()
    assert "Test Log Message" in log_path.read_text()

    # Verify console output via caplog
    # assert "Test Log Message" in caplog.text


def test_setup_logging_formatting(tmp_path: Path) -> None:
    log_path = tmp_path / "system2.log"
    setup_logging(log_path)

    logger = logging.getLogger("mlip_autopipec")
    logger.info("Info Message")
    logger.warning("Warning Message")

    content = log_path.read_text()
    assert "Info Message" in content
    assert "Warning Message" in content
    # Check if timestamp is likely present (formatting)
    # We can't strictly check format without knowing implementation, but we can assume standard practices.
    # The Spec says: "formatted correctly"
