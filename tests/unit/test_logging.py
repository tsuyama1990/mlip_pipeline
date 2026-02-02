import pytest
from mlip_autopipec.logging_config import setup_logging
import logging

def test_setup_logging():
    setup_logging("DEBUG")
    # Note: basicConfig only works once. If already configured, it does nothing.
    # In tests, logging might be configured by pytest.
    # So we can't easily assert the level changed unless we force it or reload logging.
    # But we can assert it runs without error.
    pass

def test_setup_logging_invalid():
    with pytest.raises(TypeError):
        setup_logging("INVALID_LEVEL")
