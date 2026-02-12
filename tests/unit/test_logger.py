import logging
from pathlib import Path

from mlip_autopipec.core.logger import setup_logging


def test_setup_logging(tmp_path: Path) -> None:
    # Clear existing handlers to force setup
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    root_logger.handlers = []

    try:
        log_dir = tmp_path / "logs"
        setup_logging(log_dir)

        assert log_dir.exists()
        log_file = log_dir / "mlip_pipeline.log"
        assert log_file.exists()

        # Verify handlers added
        assert len(root_logger.handlers) >= 2

        # Verify content
        logging.info("Test log message")
        # Flush might be needed?

        content = log_file.read_text()
        assert "Test log message" in content

    finally:
        # Restore original handlers
        root_logger.handlers = original_handlers
