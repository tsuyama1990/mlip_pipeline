import logging

import pytest

from mlip_autopipec.utils.logging import setup_logging


def test_setup_logging_valid() -> None:
    setup_logging(level="DEBUG")
    assert logging.getLogger().getEffectiveLevel() == logging.DEBUG


def test_setup_logging_invalid() -> None:
    with pytest.raises(TypeError):
        setup_logging(level="INVALID")
