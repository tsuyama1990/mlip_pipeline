from pathlib import Path

import pytest
from ase import Atoms


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Fixture to provide a clean temporary directory."""
    return tmp_path
    # Cleanup if needed, but tmp_path is usually handled by pytest


@pytest.fixture
def mock_atoms() -> Atoms:
    """Fixture to provide a simple ASE Atoms object."""
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
