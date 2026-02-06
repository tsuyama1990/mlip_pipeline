import logging
from pathlib import Path
from mlip_autopipec.utils import setup_logging

def test_setup_logging(tmp_path: Path) -> None:
    log_file = tmp_path / "test.log"
    setup_logging(log_file=log_file)
    logger = logging.getLogger()
    logger.info("Test log message")
    assert log_file.exists()
    content = log_file.read_text()
    assert "Test log message" in content
    logger.handlers = []
