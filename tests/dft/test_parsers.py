from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.core.exceptions import DFTConvergenceError, DFTRuntimeError
from mlip_autopipec.dft.parsers import parse_pw_output


def test_parse_pw_output_success(tmp_path: Path) -> None:
    output_file = tmp_path / "pw.out"
    output_file.write_text("JOB DONE.")

    # Mock ase.io.read
    with patch("ase.io.read") as mock_read:
        atoms = Atoms("H")
        atoms.calc = MagicMock()
        # Mock methods directly on atoms if no calculator attached or if parse returns Atoms with results
        atoms.get_potential_energy = MagicMock(return_value=-13.6)
        atoms.get_forces = MagicMock(return_value=np.zeros((1,3)))
        atoms.get_stress = MagicMock(return_value=np.zeros(6))
        mock_read.return_value = atoms

        result = parse_pw_output(output_file)

        assert result.energy == -13.6
        assert np.allclose(result.forces, 0.0)
        # Type check for stress existence before accessing
        assert result.stress is not None
        assert np.allclose(result.stress, 0.0)

def test_parse_pw_output_missing_job_done(tmp_path: Path) -> None:
    output_file = tmp_path / "pw.out"
    output_file.write_text("Some random output without finish marker.")

    with pytest.raises(DFTRuntimeError, match="missing 'JOB DONE'"):
        parse_pw_output(output_file)

def test_parse_pw_output_convergence_error(tmp_path: Path) -> None:
    output_file = tmp_path / "pw.out"
    output_file.write_text("convergence not achieved")

    with pytest.raises(DFTConvergenceError, match="convergence not achieved"):
        parse_pw_output(output_file)

def test_parse_pw_output_nan_forces(tmp_path: Path) -> None:
    output_file = tmp_path / "pw.out"
    output_file.write_text("JOB DONE.")

    with patch("ase.io.read") as mock_read:
        atoms = Atoms("H")
        atoms.get_potential_energy = MagicMock(return_value=-10.0)
        atoms.get_forces = MagicMock(return_value=np.array([[np.nan, 0, 0]]))
        mock_read.return_value = atoms

        with pytest.raises(DFTRuntimeError, match="Forces contain NaN"):
            parse_pw_output(output_file)
