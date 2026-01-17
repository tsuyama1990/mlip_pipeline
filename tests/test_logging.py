import pytest
import logging
from pathlib import Path
from mlip_autopipec.core.logging import setup_logging
from mlip_autopipec.exceptions import LoggingError

def test_setup_logging(tmp_path):
    log_path = tmp_path / "test.log"
    setup_logging(log_path, level="DEBUG")

    assert log_path.exists()

    # We verify via file output.
    logging.debug("This is a debug message")
    logging.info("This is an info message")

    # Check if log file contains messages
    with open(log_path, 'r') as f:
        content = f.read()
        assert "This is a debug message" in content
        assert "This is an info message" in content

def test_logging_level(tmp_path):
    log_path = tmp_path / "test_level.log"
    setup_logging(log_path, level="WARNING")

    logging.info("Should not see this")
    logging.warning("Should see this")

    with open(log_path, 'r') as f:
        content = f.read()
        assert "Should not see this" not in content
        assert "Should see this" in content

def test_invalid_logging_level(tmp_path):
    log_path = tmp_path / "test_invalid.log"
    setup_logging(log_path, level="INVALID_LEVEL")

    # Should default to INFO (or whatever logic handles it, but not crash)
    logging.info("Should see this")

    with open(log_path, 'r') as f:
        content = f.read()
        assert "Should see this" in content

def test_logging_permission_error(tmp_path):
    # Simulate a read-only directory
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()

    # This is tricky to test portably (root vs user), but we can try chmod
    import os
    os.chmod(ro_dir, 0o500) # Read/Execute only, no Write

    try:
        log_path = ro_dir / "test.log"

        # Should raise LoggingError as per our implementation (inherits from MLIPError)
        with pytest.raises(LoggingError):
            setup_logging(log_path)
    finally:
        os.chmod(ro_dir, 0o700) # Restore permissions for cleanup
