# ruff: noqa: N999
# FIXME: The above comment is a temporary workaround for a ruff bug.
# It should be removed once the bug is fixed.
# For more information, see: https://github.com/astral-sh/ruff/issues/10515
"""
Unit and integration tests for the DFTFactory.

This module contains a suite of tests to validate the functionality of the
`DFTFactory` class, ensuring its reliability and correctness.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest
from ase.build import bulk

# Correctly import the DFTFactory
from mlip_autopipec.modules.dft import DFTFactory


@pytest.fixture
def dft_factory(tmp_path):
    """Fixture to create a DFTFactory instance for testing."""
    # Create a dummy executable for testing purposes
    qe_executable = tmp_path / "pw.x"
    qe_executable.touch()
    qe_executable.chmod(0o755)
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    return DFTFactory(
        qe_executable_path=str(qe_executable),
        pseudo_dir=pseudo_dir,
    )


def test_dft_factory_initialization(dft_factory):
    """Test that the DFTFactory initializes correctly."""
    assert dft_factory is not None
    assert dft_factory.max_retries == 3


def test_get_heuristic_parameters(dft_factory):
    """Test the heuristic parameter generation for a simple case."""
    atoms = bulk("Si", "diamond", a=5.43)
    params = dft_factory._get_heuristic_parameters(atoms)

    assert params.cutoffs.wavefunction == 35.0
    assert params.cutoffs.density == 280.0
    assert params.k_points == (2, 2, 2)
    assert params.magnetism is None


def test_get_heuristic_parameters_magnetic(dft_factory):
    """Test heuristic parameter generation for a magnetic element."""
    atoms = bulk("Fe", "bcc", a=2.87)
    params = dft_factory._get_heuristic_parameters(atoms)

    assert params.magnetism is not None
    assert params.magnetism.nspin == 2
    assert params.magnetism.starting_magnetization["Fe"] == 0.5


@patch("subprocess.run")
@patch("mlip_autopipec.modules.dft.ase_read")
def test_dft_factory_run_successful(
    mock_ase_read,
    mock_subprocess_run,
    dft_factory,
):
    """Test a successful run of the DFTFactory."""
    # Mock the subprocess to simulate a successful run
    mock_subprocess_run.return_value = MagicMock(
        returncode=0,
        stdout="OK",
        stderr="",
    )

    # Mock the ASE read function to return a dummy Atoms object with results
    mock_atoms = bulk("Si", "diamond", a=5.43)
    mock_atoms.info["energy"] = -100.0
    mock_atoms.info["forces"] = [[0.0, 0.0, 0.0]] * 2
    mock_atoms.info["stress"] = [0.0] * 6
    mock_ase_read.return_value = mock_atoms

    atoms = bulk("Si", "diamond", a=5.43)
    result = dft_factory.run(atoms)

    assert result is not None
    assert result.energy == -100.0
    assert len(result.forces) == 2


@patch("subprocess.run")
def test_dft_factory_run_retry_and_succeed(mock_subprocess_run, dft_factory):
    """Test the retry mechanism for a recoverable error."""
    # Simulate a failure on the first attempt, then success
    failed_run = subprocess.CalledProcessError(1, "pw.x")
    failed_run.stdout = "convergence NOT achieved"
    failed_run.stderr = ""
    mock_subprocess_run.side_effect = [
        failed_run,
        MagicMock(returncode=0, stdout="OK", stderr=""),
    ]

    # Mock the ASE read function for the successful run
    mock_atoms = bulk("Si", "diamond", a=5.43)
    mock_atoms.info["energy"] = -100.0
    mock_atoms.info["forces"] = [[0.0, 0.0, 0.0]] * 2
    mock_atoms.info["stress"] = [0.0] * 6
    with patch(
        "mlip_autopipec.modules.dft.ase_read",
        return_value=mock_atoms,
    ):
        atoms = bulk("Si", "diamond", a=5.43)
        result = dft_factory.run(atoms)

    assert result is not None
    assert mock_subprocess_run.call_count == 2


@pytest.mark.integration
def test_dft_factory_integration_silicon(tmp_path):
    """
    Integration test for a simple silicon calculation.

    Note: This test requires a real Quantum Espresso installation.
    """
    pytest.skip("Skipping integration test; requires a real QE installation.")
    # Example of what a real integration test would look like:
    # qe_path = "/path/to/your/pw.x"
    # factory = DFTFactory(qe_executable_path=qe_path)
    # atoms = bulk("Si", "diamond", a=5.43)
    # result = factory.run(atoms)
    # assert result.energy < 0
    # assert np.allclose(result.forces, 0, atol=1e-3)
