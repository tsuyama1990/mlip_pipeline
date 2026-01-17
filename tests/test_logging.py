import logging

from mlip_autopipec.core.logging import setup_logging


def test_setup_logging_creates_file(tmp_path):
    """Test that setup_logging creates the log file."""
    # Reset logging
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()

    log_file = tmp_path / "test.log"
    setup_logging(log_file)

    assert log_file.exists()

    # Test logging writes to file
    logging.info("Test log message")

    # Force flush handlers to ensure data is written
    for handler in logging.getLogger().handlers:
        handler.flush()

    with log_file.open("r") as f:
        content = f.read()
        assert "Test log message" in content

def test_setup_logging_creates_parent_dir(tmp_path):
    """Test that setup_logging creates parent directories."""
    log_file = tmp_path / "subdir" / "test.log"
    setup_logging(log_file)
    assert log_file.parent.exists()
