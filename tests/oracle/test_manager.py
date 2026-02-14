"""Tests for DFT Manager."""

from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from pyacemaker.core.config import DFTConfig
from pyacemaker.oracle.manager import DFTManager


@pytest.fixture
def config(tmp_path: Path) -> DFTConfig:
    """Return a default DFT configuration."""
    # Create dummy pseudopotential file
    pp_file = tmp_path / "H.pbe.UPF"
    pp_file.touch()
    return DFTConfig(pseudopotentials={"H": str(pp_file)}, max_retries=3)


def test_compute_success(config: DFTConfig) -> None:
    """Test successful DFT calculation."""
    manager = DFTManager(config)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])

    with patch("ase.Atoms.get_potential_energy", return_value=-10.0) as mock_get_pe:
        result = manager.compute(atoms)
        assert result is not None
        assert result.calc is not None
        mock_get_pe.assert_called_once()
        # Ensure result is a new object (copy) to avoid side effects
        assert result is not atoms
        assert atoms.calc is None


def test_compute_failure_retry_recoverable(config: DFTConfig) -> None:
    """Test DFT calculation with retry on recoverable error."""
    manager = DFTManager(config)
    atoms = Atoms("H2")

    # Fail once with recoverable error, then succeed
    error_msg = "Error: SCF not converged after 100 iterations"
    with (
        patch(
            "ase.Atoms.get_potential_energy", side_effect=[Exception(error_msg), -10.0]
        ) as mock_get_pe,
        patch("pyacemaker.oracle.manager.create_calculator") as mock_create_calc,
    ):
        result = manager.compute(atoms)
        assert result is not None
        assert mock_get_pe.call_count == 2
        assert mock_create_calc.call_count == 2

        # Check arguments to create_calculator
        args_list = mock_create_calc.call_args_list
        assert args_list[0][0][1] == 0  # attempt 0
        assert args_list[1][0][1] == 1  # attempt 1


def test_compute_failure_fatal(config: DFTConfig) -> None:
    """Test DFT calculation failing immediately on fatal error."""
    manager = DFTManager(config)
    atoms = Atoms("H2")

    # Fatal error
    error_msg = "Syntax error in input file"
    with patch("ase.Atoms.get_potential_energy", side_effect=Exception(error_msg)) as mock_get_pe:
        result = manager.compute(atoms)
        assert result is None
        # Should NOT retry
        assert mock_get_pe.call_count == 1


def test_compute_failure_permanent(config: DFTConfig) -> None:
    """Test DFT calculation failing all retries (recoverable but persistent)."""
    manager = DFTManager(config)
    atoms = Atoms("H2")

    error_msg = "SCF not converged"
    with patch("ase.Atoms.get_potential_energy", side_effect=Exception(error_msg)) as mock_get_pe:
        result = manager.compute(atoms)
        assert result is None
        # Should retry max_retries times
        assert mock_get_pe.call_count == config.max_retries


def test_compute_batch(config: DFTConfig) -> None:
    """Test batch computation (generator)."""
    manager = DFTManager(config)
    atoms_list = [Atoms("H"), Atoms("He")]

    # Mock get_potential_energy for both atoms
    with patch("ase.Atoms.get_potential_energy", side_effect=[-1.0, -2.0]):
        # Now returns an iterator
        results_iter = manager.compute_batch(atoms_list)
        results = list(results_iter)
        assert len(results) == 2
        assert results[0] is not None
        assert results[1] is not None
