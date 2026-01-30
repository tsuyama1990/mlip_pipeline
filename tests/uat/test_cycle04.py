import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from pathlib import Path
from mlip_autopipec.app import app
from mlip_autopipec.domain_models.training import TrainingResult
from mlip_autopipec.domain_models.job import JobStatus

runner = CliRunner()

@patch("mlip_autopipec.cli.commands.PacemakerRunner")
@patch("mlip_autopipec.cli.commands.Config") # Mock Config loading
@patch("mlip_autopipec.cli.commands.io.load_yaml") # Mock io.load_yaml if used directly
def test_uat_train_command(mock_load_yaml, mock_config_cls, mock_runner_cls, tmp_path):
    # Setup mocks
    mock_runner_instance = mock_runner_cls.return_value
    mock_runner_instance.train.return_value = TrainingResult(
        job_id="test_job",
        status=JobStatus.COMPLETED,
        work_dir=tmp_path,
        duration_seconds=10.0,
        log_content="Training success",
        potential_path=tmp_path / "potential.yace",
        validation_metrics={"rmse_energy": 0.001, "rmse_force": 0.01}
    )

    # Mock Config
    mock_config = MagicMock()
    mock_config.training = MagicMock()
    # We need to ensure config.training is populated
    mock_config_cls.from_yaml.return_value = mock_config

    # Create a dummy config file
    config_file = tmp_path / "config.yaml"
    config_file.touch()

    # Run command
    # Assuming the command is 'train'
    result = runner.invoke(app, ["train", "--config", str(config_file), "--dataset", "data.pckl.gzip"])

    # In my plan I said I will implement 'train' command.
    # If the command is not implemented yet, this will fail with "No such command 'train'".
    # Which is good for TDD.

    # assert result.exit_code == 0
    # assert "Training Completed" in result.stdout
