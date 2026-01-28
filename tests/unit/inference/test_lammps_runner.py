from unittest.mock import patch

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.inference import BaselinePotential, InferenceConfig
from mlip_autopipec.inference.runner import LammpsRunner


@pytest.fixture
def mock_atoms():
    return Atoms("Al2", positions=[[0, 0, 0], [2, 0, 0]], cell=[10, 10, 10], pbc=True)


@pytest.fixture
def base_config(tmp_path):
    return InferenceConfig(
        lammps_executable=tmp_path / "lmp",
        temperature=500.0,
        steps=2000,
        uncertainty_threshold=4.5,
        baseline_potential=BaselinePotential.ZBL,
        timestep=0.001
    )


@patch("mlip_autopipec.inference.runner.ScriptGenerator")
@patch("mlip_autopipec.inference.runner.subprocess.run")
@patch("mlip_autopipec.inference.runner.LammpsLogParser")
def test_run_halted_execution(mock_parser, mock_run, mock_script_gen, base_config, mock_atoms, tmp_path):
    # Setup
    potential_path = tmp_path / "pot.yace"
    potential_path.touch()

    # Mock Script Generator
    mock_script_gen_instance = mock_script_gen.return_value
    mock_script_gen_instance.generate.return_value = "mock_script_content"

    # Mock subprocess failure due to halt (LAMMPS often exits with error on halt)
    mock_run.return_value.returncode = 1

    # Mock Log Parser finding halt
    mock_parser.parse.return_value = (6.0, True, 150)

    runner = LammpsRunner(base_config, tmp_path)
    result = runner.run(mock_atoms, potential_path, "halted_uid")

    assert result.succeeded is True  # Halted is considered a successful "detection"
    assert result.halted is True
    assert result.max_gamma_observed == 6.0
    assert result.halt_step == 150

    # Verify ScriptGenerator usage
    mock_script_gen.assert_called_with(base_config)
    mock_script_gen_instance.generate.assert_called()


@patch("mlip_autopipec.inference.runner.ScriptGenerator")
@patch("mlip_autopipec.inference.runner.subprocess.run")
@patch("mlip_autopipec.inference.runner.LammpsLogParser")
def test_run_normal_execution(mock_parser, mock_run, mock_script_gen, base_config, mock_atoms, tmp_path):
    potential_path = tmp_path / "pot.yace"
    potential_path.touch()

    mock_script_gen.return_value.generate.return_value = "mock_script"

    mock_run.return_value.returncode = 0
    mock_parser.parse.return_value = (2.0, False, None)

    runner = LammpsRunner(base_config, tmp_path)
    result = runner.run(mock_atoms, potential_path, "normal_uid")

    assert result.succeeded is True
    assert result.halted is False
    assert result.max_gamma_observed == 2.0
