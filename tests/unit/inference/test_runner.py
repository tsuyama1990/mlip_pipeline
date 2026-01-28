from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.inference.runner import LammpsRunner


@pytest.fixture
def mock_config(tmp_path):
    # Ensure command has no spaces for safety check if simple string
    return InferenceConfig(lammps_executable=tmp_path / "lmp", temperature=300, timestep=0.001, steps=100)


@pytest.fixture
def mock_atoms():
    # LAMMPS requires a cell
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=True)


@patch("mlip_autopipec.inference.runner.subprocess.run")
@patch("mlip_autopipec.inference.runner.LammpsLogParser")
def test_runner_success(mock_parser, mock_run, mock_config, mock_atoms, tmp_path):
    # Setup mocks
    mock_run.return_value.returncode = 0
    mock_parser.parse.return_value = (1.5, False, None)

    # Mock existence of potential
    potential_path = tmp_path / "pot.yace"
    potential_path.touch()

    runner = LammpsRunner(mock_config, tmp_path)
    result = runner.run(mock_atoms, potential_path, uid="test_uid")

    assert result.succeeded is True
    assert result.max_gamma_observed == 1.5
    assert result.uid == "test_uid"


def test_runner_input_validation(mock_config, tmp_path):
    runner = LammpsRunner(mock_config, tmp_path)
    with pytest.raises(TypeError):
        runner.run("not atoms", Path("pot.yace"), "uid")

    atoms = Atoms("H", cell=[10,10,10], pbc=True)
    with pytest.raises(FileNotFoundError):
        runner.run(atoms, Path("nonexistent.yace"), "uid")
