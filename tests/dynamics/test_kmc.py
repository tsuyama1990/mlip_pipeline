"""Tests for EON wrapper."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from pyacemaker.core.config import EONConfig
from pyacemaker.dynamics.kmc import EONWrapper


@pytest.fixture
def mock_atoms() -> Atoms:
    """Create a mock Atoms object."""
    return Atoms("Si2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[3, 3, 3], pbc=True)


@pytest.fixture
def eon_config() -> EONConfig:
    """Create a mock EON configuration."""
    return EONConfig(executable="mock_eon")


def test_eon_run_search(mock_atoms: Atoms, eon_config: EONConfig, tmp_path: Path) -> None:
    """Test EON search execution."""
    wrapper = EONWrapper(eon_config)
    potential_path = Path("potential.yace")

    # Need to mock subprocess.run inside run_search
    with (
        patch("pyacemaker.dynamics.kmc.subprocess.run") as mock_run,
        patch("pyacemaker.dynamics.kmc.shutil.which") as mock_which,
    ):
        mock_run.return_value.returncode = 0
        mock_which.return_value = "/usr/bin/mock_eon"

        wrapper.run_search(mock_atoms, potential_path, work_dir=tmp_path)

        # Check if subprocess was called with correct command (we'll implement this later)
        # For now, it won't be called because run_search is pass
        # So this test will FAIL, which is good for TDD.
        mock_run.assert_called_once()

        # Also check files
        assert (tmp_path / "config.ini").exists()
        assert (tmp_path / "pace_driver.py").exists()


def test_eon_failure(mock_atoms: Atoms, eon_config: EONConfig, tmp_path: Path) -> None:
    """Test EON failure handling."""
    wrapper = EONWrapper(eon_config)
    potential_path = Path("potential.yace")

    with (
        patch("pyacemaker.dynamics.kmc.subprocess.run") as mock_run,
        patch("pyacemaker.dynamics.kmc.shutil.which") as mock_which,
    ):
        mock_run.side_effect = subprocess.CalledProcessError(1, ["mock_eon"])
        mock_which.return_value = "/usr/bin/mock_eon"

        with pytest.raises(RuntimeError, match="EON execution failed"):
            wrapper.run_search(mock_atoms, potential_path, work_dir=tmp_path)
