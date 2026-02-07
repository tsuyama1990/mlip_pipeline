import logging

import pytest

from mlip_autopipec.utils.logging import setup_logging


def test_setup_logging_valid() -> None:
    setup_logging("DEBUG")
    assert logging.getLogger().getEffectiveLevel() == logging.DEBUG

    setup_logging("WARNING")
    assert logging.getLogger().getEffectiveLevel() == logging.WARNING


def test_setup_logging_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid log level"):
        setup_logging("INVALID_LEVEL")
