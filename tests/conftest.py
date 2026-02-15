"""Global pytest configuration."""

import pytest
from pathlib import Path

from pyacemaker.core.config import CONSTANTS

@pytest.fixture(autouse=True)
def skip_file_security_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip file security checks for tests running in /tmp."""
    # By default, tests use tmp_path which is outside CWD (/app).
    # We must allow this for tests to pass.
    # Security tests should explicitly override this or use valid paths.
    monkeypatch.setattr(CONSTANTS, "skip_file_checks", True)
