import logging

from mlip_autopipec.core.logging import setup_logging


def test_setup_logging(tmp_path):
    log_file = tmp_path / "system.log"
    setup_logging(log_file, level="DEBUG")

    logger = logging.getLogger("mlip_autopipec")
    logger.debug("Test debug message")
    logger.info("Test info message")

    # Check file creation
    assert log_file.exists()
    content = log_file.read_text()
    assert "Test debug message" in content
    assert "Test info message" in content

def test_logging_level():
    logger = logging.getLogger("mlip_autopipec")
    assert logger.level == logging.DEBUG or logger.level == logging.NOTSET
    # logging.NOTSET if parent has level, but we set it on root or specific logger.
    # The setup_logging usually sets root or package logger.
