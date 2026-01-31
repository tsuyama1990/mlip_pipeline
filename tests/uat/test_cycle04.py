from unittest.mock import patch

from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.domain_models.config import Config, TrainingConfig, PotentialConfig, BulkStructureGenConfig, MDConfig
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.training import TrainingResult

runner = CliRunner()

@patch("mlip_autopipec.cli.commands.DatasetManager")
@patch("mlip_autopipec.cli.commands.PacemakerRunner")
@patch("mlip_autopipec.cli.commands.io.load_structures")
@patch("mlip_autopipec.domain_models.config.Config.from_yaml")
def test_train_command(mock_config_load, mock_load_structs, mock_runner_cls, mock_dataset_cls, tmp_path):
    # Setup Config
    config = Config(
        project_name="test",
        potential=PotentialConfig(elements=["Si"], cutoff=5.0),
        structure_gen=BulkStructureGenConfig(strategy="bulk", element="Si", crystal_structure="diamond", lattice_constant=5.43),
        md=MDConfig(temperature=300, n_steps=100, ensemble="NVT"),
        training=TrainingConfig(batch_size=10),
    )
    mock_config_load.return_value = config

    # Setup mocks
    mock_runner = mock_runner_cls.return_value
    mock_runner.train.return_value = TrainingResult(
        job_id="train_01",
        status=JobStatus.COMPLETED,
        work_dir=tmp_path,
        duration_seconds=1.0,
        log_content="RMSE: 0.01",
        potential_path=tmp_path / "potential.yace",
        validation_metrics={"energy": 0.01}
    )

    # Create dummy config file
    config_path = tmp_path / "config.yaml"
    config_path.touch()

    # Create dummy structures file
    structures_path = tmp_path / "structures.xyz"
    structures_path.touch()

    # Invoke command
    # Assuming the command will be `mlip-auto train --config ... --dataset ...`
    result = runner.invoke(app, ["train", "--config", str(config_path), "--dataset", str(structures_path)])

    assert result.exit_code == 0
    assert "Training Completed" in result.stdout

    mock_runner.train.assert_called_once()
    mock_dataset_cls.return_value.convert.assert_called_once()
