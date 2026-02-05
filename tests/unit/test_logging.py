import logging
from utils.logging import setup_logging
from pathlib import Path

def test_setup_logging(tmp_path: Path) -> None:
    log_file = tmp_path / "test.log"
    logger = setup_logging("DEBUG", str(log_file))

    assert logger.name == "mlip_pipeline"
    assert logging.getLogger().getEffectiveLevel() == logging.DEBUG

    logger.info("Test message")

    assert log_file.exists()
    content = log_file.read_text()
    assert "Test message" in content
