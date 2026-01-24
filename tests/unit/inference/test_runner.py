from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.inference.runner import LammpsRunner


@pytest.fixture
def mock_config(tmp_path):
    return InferenceConfig(
        lammps_executable=tmp_path / "lmp",
        steps=100,
        uncertainty_threshold=5.0
    )

@pytest.fixture
def mock_atoms():
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])

@patch("mlip_autopipec.inference.runner.subprocess.run")
@patch("mlip_autopipec.inference.runner.shutil.which")
@patch("mlip_autopipec.inference.runner.LammpsInputWriter")
@patch("mlip_autopipec.inference.runner.LogParser")
def test_runner_success(mock_parser, mock_writer, mock_which, mock_run, mock_config, mock_atoms, tmp_path):
    # Setup mocks
    mock_which.return_value = "/usr/bin/lmp"
    mock_run.return_value.returncode = 0

    # Mock Writer returning paths
    input_file = tmp_path / "in.lammps"
    data_file = tmp_path / "data.lammps"
    log_file = tmp_path / "log.lammps"
    dump_file = tmp_path / "dump.out"
    # Ensure they have names
    input_file.touch()

    mock_writer_instance = mock_writer.return_value
    mock_writer_instance.write_inputs.return_value = (input_file, data_file, log_file, dump_file)

    # Mock Parser
    # max_gamma, halted, halt_step
    mock_parser.parse.return_value = (1.5, False, None)

    runner = LammpsRunner(mock_config, tmp_path)
    result = runner.run(mock_atoms, Path("pot.yace"))

    assert result.succeeded is True
    assert result.max_gamma_observed == 1.5
    assert len(result.uncertain_structures) == 0

    # Check subprocess call
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert args[0][0] == "/usr/bin/lmp"
    assert args[0][2] == "in.lammps"
    assert kwargs['shell'] is False

@patch("mlip_autopipec.inference.runner.subprocess.run")
@patch("mlip_autopipec.inference.runner.shutil.which")
@patch("mlip_autopipec.inference.runner.LammpsInputWriter")
@patch("mlip_autopipec.inference.runner.LogParser")
def test_runner_halted(mock_parser, mock_writer, mock_which, mock_run, mock_config, mock_atoms, tmp_path):
    mock_which.return_value = "/usr/bin/lmp"

    # Simulate return code 1 which usually happens on fix halt error hard?
    # Or 0 if handled gracefully.
    # If "error hard", LAMMPS returns 1.
    mock_run.return_value.returncode = 1
    mock_run.return_value.stderr = ""

    input_file = tmp_path / "in.lammps"
    data_file = tmp_path / "data.lammps"
    log_file = tmp_path / "log.lammps"
    dump_file = tmp_path / "dump.out"
    input_file.touch()
    dump_file.write_text("dummy dump content")

    mock_writer.return_value.write_inputs.return_value = (input_file, data_file, log_file, dump_file)

    # Mock Parser saying it halted
    mock_parser.parse.return_value = (6.0, True, 50)

    runner = LammpsRunner(mock_config, tmp_path)
    result = runner.run(mock_atoms, Path("pot.yace"))

    # Even if return code is 1, if parser detects halt, it might be considered "succeeded" in terms of "we ran it"
    # BUT typically if it halts due to uncertainty, 'succeeded' might mean "completed without crash" OR "completed the active learning step".
    # The UAT says: "Return object indicates halted=True".
    # InferenceResult definition has `succeeded`.
    # If it halted, did the simulation succeed? Technically it finished early.
    # Usually we want `succeeded=True` (as in, no system crash) but maybe a flag for `halted`.
    # Current InferenceResult has: succeeded, max_gamma_observed, uncertain_structures.
    # It does NOT have 'halted' field.
    # However, uncertain_structures being non-empty implies it found uncertainty.

    # If return code is 1, existing runner logic returns succeeded=False immediately.
    # I need to modify runner logic to check log if return code != 0, because "fix halt ... error hard" causes non-zero exit.

    assert result.max_gamma_observed == 6.0
    assert len(result.uncertain_structures) == 1
    assert result.uncertain_structures[0] == dump_file
