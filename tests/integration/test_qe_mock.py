"""
Integration tests for QERunner with mocked subprocess.
"""
import pytest
import subprocess
from pathlib import Path
from ase import Atoms
from mlip_autopipec.core.config import DFTConfig
from mlip_autopipec.dft.qe_runner import QERunner
from mlip_autopipec.core.models import DFTResult
import numpy as np

def test_qe_runner_mock_execution(tmp_path, mocker):
    """Test QERunner execution with mocked binary."""
    run_dir = tmp_path / "run"
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()

    config = DFTConfig(
        command="pw.x",
        pseudopotential_dir=pseudo_dir
    )

    atoms = Atoms("H", positions=[[0,0,0]], cell=[10,10,10], pbc=True)

    runner = QERunner(config)

    # Mock subprocess
    def side_effect(args, cwd=None, stdout=None, **kwargs):
        # Write "JOB DONE" to stdout file to satisfy parser check
        if stdout:
             stdout.write("JOB DONE.")
        return subprocess.CompletedProcess(args, returncode=0)

    mocker.patch("subprocess.run", side_effect=side_effect)

    # Mock ASE read to avoid needing perfect PW output string
    # We mock 'mlip_autopipec.dft.parsers.read' because parsers.py imports it as 'read'
    mock_atoms = Atoms("H", positions=[[0,0,0]])
    from ase.calculators.singlepoint import SinglePointCalculator
    mock_atoms.calc = SinglePointCalculator(
        mock_atoms,
        energy=-1360.5698,
        forces=np.zeros((1,3)),
        stress=np.zeros(6)
    )

    mocker.patch("mlip_autopipec.dft.parsers.read", return_value=mock_atoms)

    result = runner.run_static_calculation(atoms, run_dir)

    # 100 Ry approx 1360.5 eV.
    assert result.energy == pytest.approx(-1360.5698, abs=1.0)
    assert result.forces.shape == (1, 3)
