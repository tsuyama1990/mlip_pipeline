"""Shared fixtures for tests."""

from pathlib import Path

import pytest
from ase import Atoms


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory that is cleaned up after test."""
    return tmp_path
    # tmp_path is automatically cleaned up by pytest, but we can do extra cleanup if needed.


@pytest.fixture
def sample_ase_atoms() -> Atoms:
    """Provide a sample ASE Atoms object."""
    return Atoms(
        "Si2",
        positions=[[0, 0, 0], [1.3, 1.3, 1.3]],
        cell=[5.43, 5.43, 5.43],
        pbc=True
    )
