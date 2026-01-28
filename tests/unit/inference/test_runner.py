from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.inference.runner import LammpsRunner


@pytest.fixture
def mock_config(tmp_path):
    return InferenceConfig(lammps_executable=tmp_path / "lmp", steps=100, uncertainty_threshold=5.0)


@pytest.fixture
def mock_atoms():
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])


@patch("mlip_autopipec.inference.runner.subprocess.run")
@patch("mlip_autopipec.inference.runner.shutil.which")
@patch("mlip_autopipec.inference.runner.LammpsInputWriter")
@patch("mlip_autopipec.inference.runner.LogParser")
def test_runner_success(
    mock_parser, mock_writer, mock_which, mock_run, mock_config, mock_atoms, tmp_path
):
    # Setup mocks
    mock_which.return_value = "/usr/bin/lmp"
    mock_run.return_value.returncode = 0

    input_file = tmp_path / "in.lammps"
    data_file = tmp_path / "data.lammps"
    log_file = tmp_path / "log.lammps"
    dump_file = tmp_path / "dump.out"
    input_file.touch()

    # Mock existence of potential
    potential_path = tmp_path / "pot.yace"
    potential_path.touch()

    mock_writer_instance = mock_writer.return_value
    mock_writer_instance.write_inputs.return_value = (input_file, data_file, log_file, dump_file)
    mock_parser.parse.return_value = (1.5, False, None)

    runner = LammpsRunner(mock_config, tmp_path)
    result = runner.run(mock_atoms, potential_path)

    assert result.succeeded is True
    assert result.max_gamma_observed == 1.5
    assert len(result.uncertain_structures) == 0


def test_runner_input_validation(mock_config, tmp_path):
    runner = LammpsRunner(mock_config, tmp_path)
    with pytest.raises(TypeError):
        runner.run("not atoms", Path("pot.yace"))

    atoms = Atoms("H")
    with pytest.raises(FileNotFoundError):
        runner.run(atoms, Path("nonexistent.yace"))


@patch("mlip_autopipec.inference.runner.shutil.which")
def test_runner_executable_resolution(mock_which, mock_config, tmp_path):
    # Test valid
    mock_which.return_value = "/bin/lmp"
    runner = LammpsRunner(mock_config, tmp_path)
    assert runner._resolve_executable() == "/bin/lmp"

    # Test not found
    mock_which.return_value = None
    with pytest.raises(RuntimeError):
        runner._resolve_executable()

    # Test fallback
    mock_config.lammps_executable = None
    # Allow looking up by name OR by resolved path
    mock_which.side_effect = lambda x: "/bin/lmp_mpi" if x in ["lmp_mpi", "/bin/lmp_mpi"] else None

    runner = LammpsRunner(mock_config, tmp_path)
    assert runner._resolve_executable() == "/bin/lmp_mpi"
