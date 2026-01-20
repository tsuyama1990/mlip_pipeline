"""
Tests for parsing logic.
"""
import pytest
from pathlib import Path
from mlip_autopipec.dft import parsers
from mlip_autopipec.core.exceptions import DFTRuntimeError, DFTConvergenceError
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np

def test_parse_pw_output_success(tmp_path, mocker):
    """Test successful parsing."""
    output_file = tmp_path / "pw.out"
    output_file.write_text("JOB DONE.\n")

    # Mock ASE read
    atoms = Atoms("Si")
    calc = SinglePointCalculator(
        atoms,
        energy=-100.0,
        forces=np.zeros((1,3)),
        stress=np.zeros(6)
    )
    atoms.calc = calc

    mocker.patch("mlip_autopipec.dft.parsers.read", return_value=atoms)

    result = parsers.parse_pw_output(output_file)
    assert result.energy == -100.0
    assert np.allclose(result.forces, 0.0)

def test_parse_pw_output_missing_job_done(tmp_path):
    """Test detection of incomplete jobs."""
    output_file = tmp_path / "pw.out"
    output_file.write_text("Some random output...\n")

    with pytest.raises(DFTRuntimeError, match="JOB DONE not found"):
        parsers.parse_pw_output(output_file)

def test_parse_pw_output_convergence_error(tmp_path):
    """Test detection of convergence failure."""
    output_file = tmp_path / "pw.out"
    output_file.write_text("convergence not achieved\n")

    with pytest.raises(DFTConvergenceError):
        parsers.parse_pw_output(output_file)

def test_parse_pw_output_nan_forces(tmp_path, mocker):
    """Test rejection of NaN forces."""
    output_file = tmp_path / "pw.out"
    output_file.write_text("JOB DONE.\n")

    atoms = Atoms("Si")
    calc = SinglePointCalculator(
        atoms,
        energy=-100.0,
        forces=np.array([[np.nan, 0, 0]]),
        stress=np.zeros(6)
    )
    atoms.calc = calc
    mocker.patch("mlip_autopipec.dft.parsers.read", return_value=atoms)

    with pytest.raises(DFTRuntimeError, match="Forces contain NaN"):
        parsers.parse_pw_output(output_file)
