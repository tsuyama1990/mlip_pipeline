# ruff: noqa: D101, D102
from unittest.mock import MagicMock, patch

import pytest
import yaml
from ase import Atoms
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.config_schemas import SystemConfig

runner = CliRunner()


@pytest.fixture
def mock_config_file(tmp_path):
    """Creates a temporary mock YAML config file for testing."""
    config_data = {
        "target_system": {
            "elements": ["Cu"],
            "composition": {"Cu": 1.0},
        },
        "simulation_goal": "melt_quench",
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    return config_file


@patch("mlip_autopipec.app.expand_config")
@patch("mlip_autopipec.app.LammpsRunner")
@patch("mlip_autopipec.app.MagicMock")
def test_active_learning_loop_logic(
    mock_magic_mock: MagicMock,
    mock_lammps_runner: MagicMock,
    mock_expand_config: MagicMock,
    mock_config_file,
):
    """Test the main active learning loop orchestration in the CLI app.

    This integration test verifies that the `run` command correctly sequences
    the calls to the various modules when a high-uncertainty structure is
    encountered.
    """
    # Mock the LammpsRunner to yield a stable state then an Atoms object
    mock_runner_instance = MagicMock()
    mock_runner_instance.run.return_value = iter(["stable", Atoms("X")])
    mock_lammps_runner.return_value = mock_runner_instance

    # Mock the expand_config to return a valid SystemConfig with nested inference mock
    mock_system_config = MagicMock(spec=SystemConfig)
    mock_inference_params = MagicMock()
    mock_inference_params.total_simulation_steps = 10
    mock_system_config.inference = mock_inference_params
    mock_expand_config.return_value = mock_system_config

    # Instantiate mock modules that will be created inside the app
    mock_db_manager = MagicMock()
    mock_dft_runner = MagicMock()
    mock_trainer = MagicMock()
    mock_magic_mock.side_effect = [mock_db_manager, mock_dft_runner, mock_trainer]

    # Run the CLI command
    result = runner.invoke(app, ["--config", str(mock_config_file)])
    assert result.exit_code == 0

    # Assert that the modules were called in the correct order and frequency
    # The trainer should be called at the start of each cycle. The loop breaks
    # after the first uncertainty, so it should run twice.
    assert mock_trainer.train.call_count == 2
    mock_dft_runner.run.assert_called_once()
    mock_db_manager.write_calculation.assert_called_once()


def test_cli_handles_missing_file():
    """Test that the CLI exits gracefully if the config file is not found."""
    result = runner.invoke(app, ["--config", "non_existent_file.yaml"])
    assert result.exit_code != 0
    assert "Invalid value" in result.stderr
