import logging

from mlip_autopipec.core.logging import setup_logging


def test_setup_logging(caplog):
    caplog.set_level(logging.DEBUG)
    setup_logging(level="DEBUG")

    logger = logging.getLogger("test_logger")
    logger.debug("Test debug message")
    logger.info("Test info message")

    assert "Test debug message" in caplog.text
    assert "Test info message" in caplog.text
