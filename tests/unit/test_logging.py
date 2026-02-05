import pytest
import logging
from mlip_autopipec.utils.logging import setup_logging

def test_setup_logging_valid() -> None:
    """Test valid logging setup."""
    setup_logging(level="DEBUG", force=True)
    logger = logging.getLogger("test_logger")
    assert logger.getEffectiveLevel() <= logging.DEBUG

def test_setup_logging_invalid() -> None:
    """Test invalid logging level raises TypeError."""
    with pytest.raises(TypeError):
        setup_logging(level="INVALID_LEVEL")
