import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.domain_models.config import (
    Config,
    PotentialConfig,
    BulkStructureGenConfig,
    MDConfig,
    TrainingConfig
)

runner = CliRunner()


def test_train_command(tmp_path):
    """
    Scenario 4.1: The "Train" Command
    """
    # Create dummy dataset
    dset_path = tmp_path / "dataset.pckl.gzip"
    dset_path.touch()

    # Create dummy config
    config_path = tmp_path / "config.yaml"
    config_content = """
project_name: "UAT_Project"
potential:
  elements: ["Si"]
  cutoff: 5.0
  pair_style: "hybrid/overlay"
structure_gen:
  strategy: "bulk"
  element: "Si"
  crystal_structure: "diamond"
  lattice_constant: 5.43
md:
  temperature: 300.0
  n_steps: 10
  timestep: 0.001
  ensemble: "NVT"
training:
  batch_size: 10
  max_epochs: 5
"""
    config_path.write_text(config_content)

    # Mock components
    with (
        patch("mlip_autopipec.cli.commands.io.load_structures") as mock_load,
        patch("mlip_autopipec.cli.commands.DatasetManager") as mock_dm_cls,
        patch("mlip_autopipec.cli.commands.PacemakerRunner") as mock_runner_cls,
    ):
        mock_load.return_value = []  # Dummy list of structures

        mock_dm = MagicMock()
        mock_dm.convert.return_value = Path("dummy_train.pckl.gzip")
        mock_dm_cls.return_value = mock_dm

        mock_runner = MagicMock()
        # Mock successful training result
        from mlip_autopipec.domain_models.job import JobStatus
        from mlip_autopipec.domain_models.training import TrainingResult

        mock_runner.train.return_value = TrainingResult(
            job_id="test_job",
            status=JobStatus.COMPLETED,
            work_dir=Path("."),
            duration_seconds=1.0,
            log_content="ok",
            potential_path=Path("potential.yace"),
            validation_metrics={"rmse_energy": 0.001}
        )
        mock_runner_cls.return_value = mock_runner

        # Execute
        result = runner.invoke(app, ["train", "--config", str(config_path), "--dataset", str(dset_path)])

        assert result.exit_code == 0
        assert "Training Completed" in result.stdout
        assert "potential.yace" in result.stdout

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
