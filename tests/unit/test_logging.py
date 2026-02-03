import logging
from pathlib import Path

import pytest

from mlip_autopipec.logging_config import setup_logging


def test_setup_logging(temp_dir: Path) -> None:
    log_file = temp_dir / "test.log"
    setup_logging(log_level="DEBUG", log_file=log_file)

    logger = logging.getLogger("test_logger")
    logger.debug("Debug message")
    logger.info("Info message")

    # Check file output
    assert log_file.exists()
    with log_file.open("r") as f:
        content = f.read()
        assert "Debug message" in content
        assert "Info message" in content


def test_setup_logging_console(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(log_level="INFO")
    logger = logging.getLogger("console_logger")
    logger.info("Console message")
    logger.debug("Hidden debug")

    captured = capsys.readouterr()
    # logging uses StreamHandler(sys.stdout) per logging_config.py
    assert "Console message" in captured.out
    assert "Hidden debug" not in captured.out
