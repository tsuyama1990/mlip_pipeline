import logging

from mlip_autopipec.domain_models.config import LoggingConfig
from mlip_autopipec.infrastructure.logging import setup_logging


def test_setup_logging(tmp_path):
    """Test logging setup."""
    log_file = tmp_path / "test.log"
    config = LoggingConfig(level="DEBUG", file_path=log_file)

    setup_logging(config)

    logger = logging.getLogger("mlip_autopipec")
    assert logger.getEffectiveLevel() == logging.DEBUG

    # Check handlers (attached to root logger)
    root_logger = logging.getLogger()
    handlers = root_logger.handlers
    assert len(handlers) > 0
    # Should have FileHandler and RichHandler (or StreamHandler)
    has_file = any(isinstance(h, logging.FileHandler) for h in handlers)
    assert has_file

    # Verify file creation after logging something
    logger.info("Test message")
    # File might not be written immediately due to buffering, but FileHandler usually opens it.
    assert log_file.exists()

    content = log_file.read_text()
    assert "Test message" in content
