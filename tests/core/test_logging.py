"""Tests for logging configuration."""

from collections.abc import Generator

import pytest
from loguru import logger

from pyacemaker.core.logging import get_logger, setup_logging


@pytest.fixture(autouse=True)
def clean_logger() -> Generator[None, None, None]:
    """Reset logger before and after each test."""
    logger.remove()
    yield
    logger.remove()


def test_setup_logging(capsys: pytest.CaptureFixture[str]) -> None:
    """Test logging setup."""
    setup_logging("DEBUG")
    logger.debug("Test debug message")

    captured = capsys.readouterr()
    # Loguru writes to stderr by default
    assert "Test debug message" in captured.err
    # Check default name [pyacemaker]
    # The format is "[{time}] [{level}] [{extra[name]}] {message}"
    assert "[pyacemaker]" in captured.err


def test_get_logger(capsys: pytest.CaptureFixture[str]) -> None:
    """Test get_logger returns a bound logger."""
    # Ensure handler exists
    setup_logging("INFO")

    log = get_logger("TestLogger")
    log.info("Message from bound logger")

    captured = capsys.readouterr()
    assert "Message from bound logger" in captured.err
    # Check bound name [TestLogger]
    assert "[TestLogger]" in captured.err


def test_logging_propagation(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that logger names are correctly bound."""
    setup_logging("DEBUG")

    # Simulate a module logger
    module_logger = get_logger("pyacemaker.core.test_module")
    module_logger.info("Test message from module")

    captured = capsys.readouterr()
    assert "Test message from module" in captured.err
    assert "[pyacemaker.core.test_module]" in captured.err
