import logging
from pathlib import Path

from mlip_autopipec.infrastructure.logging import configure_logging


def test_logging_configuration(tmp_path: Path) -> None:
    log_file = tmp_path / "test.log"
    configure_logging(level="DEBUG", log_file=log_file)

    logger = logging.getLogger("mlip_autopipec")

    # Verify handlers
    handlers = logging.getLogger().handlers
    # Should have Console (Rich) and File
    assert any(isinstance(h, logging.FileHandler) for h in handlers)

    # Log something
    logger.info("Test message")

    # Check file content
    # Note: Logging might be async or buffered, so we might need to flush
    for h in handlers:
        h.flush()

    assert log_file.exists()
    content = log_file.read_text()
    assert "Test message" in content
