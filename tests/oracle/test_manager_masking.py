"""Tests for force masking in DFT Manager."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.config import DFTConfig
from pyacemaker.oracle.manager import DFTManager


@pytest.fixture
def config(tmp_path: Path) -> DFTConfig:
    """Return a DFT configuration with valid pseudo file."""
    pp_file = tmp_path / "H.pbe.UPF"
    pp_file.touch()
    return DFTConfig(
        pseudopotentials={"H": str(pp_file)},
        embedding_enabled=True,
        embedding_buffer=5.0
    )

def test_force_mask_initialization(config: DFTConfig) -> None:
    """Test that force mask is initialized if missing when embedding is enabled."""
    manager = DFTManager(config)
    atoms = Atoms("H", positions=[[0, 0, 0]])

    # Mock compute internals
    with (
        patch("pyacemaker.oracle.manager.create_calculator"),
        patch("ase.Atoms.get_potential_energy", return_value=-10.0),
    ):
        result = manager.compute(atoms)

        assert "force_mask" in result.arrays
        # Core atoms should be 1.0
        assert np.all(result.arrays["force_mask"] == 1.0)
        assert result.pbc.all()


def test_force_mask_preservation(config: DFTConfig) -> None:
    """Test that existing force mask is preserved."""
    manager = DFTManager(config)
    atoms = Atoms("H", positions=[[0, 0, 0]])
    mask = np.array([0.0])  # Buffer atom
    atoms.new_array("force_mask", mask)  # type: ignore[no-untyped-call]

    # Mock compute internals
    with (
        patch("pyacemaker.oracle.manager.create_calculator"),
        patch("ase.Atoms.get_potential_energy", return_value=-10.0),
    ):
        result = manager.compute(atoms)

        assert "force_mask" in result.arrays
        assert result.arrays["force_mask"][0] == 0.0
