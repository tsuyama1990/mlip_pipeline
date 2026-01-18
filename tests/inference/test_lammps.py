import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ase import Atoms
import numpy as np
from mlip_autopipec.inference.lammps_runner import LammpsRunner
from mlip_autopipec.config.schemas.inference import InferenceConfig, InferenceResult

@pytest.fixture
def mock_subprocess():
    with patch("subprocess.run") as mock:
        yield mock

@pytest.fixture
def potential_file(tmp_path):
    p = tmp_path / "model.yace"
    p.touch()
    return p

@pytest.fixture
def config(potential_file):
    return InferenceConfig(
        lammps_executable="lmp_serial",
        potential_path=potential_file,
        temperature=300,
        steps=100,
        uq_threshold=5.0
    )

def test_runner_execution_success(mock_subprocess, config):
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stdout = "LAMMPS Log"
    mock_subprocess.return_value.stderr = ""

    with patch("mlip_autopipec.inference.lammps_runner.UncertaintyChecker") as MockChecker:
        runner = LammpsRunner(config)
        MockChecker.return_value.parse_dump.return_value = []
        MockChecker.return_value.max_gamma = 1.0

        atoms = Atoms("H2", positions=[[0,0,0], [0,0,0.7]], cell=[10,10,10])
        result = runner.run(atoms)

    assert isinstance(result, InferenceResult)
    assert result.succeeded is True
    assert result.max_gamma_observed < 5.0

def test_runner_execution_failure(mock_subprocess, config):
    runner = LammpsRunner(config)
    atoms = Atoms("H2", cell=[10,10,10])

    mock_subprocess.side_effect = Exception("LAMMPS crashed")

    with pytest.raises(Exception):
        runner.run(atoms)

def test_runner_detects_uncertainty(mock_subprocess, config):
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stdout = "LAMMPS Log"
    mock_subprocess.return_value.stderr = ""

    with patch("mlip_autopipec.inference.lammps_runner.UncertaintyChecker") as MockChecker:
        runner = LammpsRunner(config)

        # Simulate finding uncertain atoms
        bad_atom = Atoms("H")
        bad_atom.arrays['gamma'] = np.array([8.0])
        bad_atom.cell = [10,10,10]

        MockChecker.return_value.parse_dump.return_value = [bad_atom]
        MockChecker.return_value.max_gamma = 8.0

        atoms = Atoms("H2", cell=[10,10,10])
        result = runner.run(atoms)

    assert result.succeeded is True
    assert len(result.uncertain_structures) == 1
    assert result.max_gamma_observed == 8.0
