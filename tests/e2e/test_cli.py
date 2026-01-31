from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.training import TrainingResult

runner = CliRunner()


def test_init(tmp_path: Path) -> None:
    """Test 'init' command."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "Created template configuration" in result.stdout
        assert Path("config.yaml").exists()


def test_init_existing(tmp_path: Path) -> None:
    """Test 'init' fails if file exists."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("config.yaml").touch()
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1
        assert "already exists" in result.stdout


def test_init_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 'init' handles exceptions during write."""
    from mlip_autopipec.infrastructure import io

    def mock_dump(*args: Any, **kwargs: Any) -> None:
        msg = "Permission denied"
        raise OSError(msg)

    monkeypatch.setattr(io, "dump_yaml", mock_dump)

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1
        assert "Failed to create config: Permission denied" in result.stdout


def test_check_valid(tmp_path: Path) -> None:
    """Test 'check' with valid config."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Create config first
        runner.invoke(app, ["init"])

        result = runner.invoke(app, ["check"])
        assert result.exit_code == 0
        assert "Configuration valid" in result.stdout
        assert Path("mlip_pipeline.log").exists()


def test_check_invalid(tmp_path: Path) -> None:
    """Test 'check' with invalid config."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        p = Path("config.yaml")
        p.write_text("project_name: 'Bad'\npotential:\n  cutoff: -1\n  elements: ['A']")

        result = runner.invoke(app, ["check"])
        assert result.exit_code == 1
        assert "Validation failed" in result.stdout
        assert "Cutoff must be greater than 0" in result.stdout

def test_train_command(tmp_path: Path) -> None:
    """Test 'train' command integration."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Setup config
        runner.invoke(app, ["init"])

        # Manually append training config as init doesn't produce it yet (optional)
        # Actually init produces a config without 'training'.
        # We need to add 'training' section.
        config_path = Path("config.yaml")
        config_content = config_path.read_text()
        config_content += "\ntraining:\n  batch_size: 10\n  max_epochs: 5\n"
        config_path.write_text(config_content)

        # Create dummy dataset
        dataset_path = Path("train.pckl.gzip")
        dataset_path.touch()

        # Mock PacemakerRunner.train
        with patch("mlip_autopipec.physics.training.pacemaker.PacemakerRunner.train") as mock_train:
            mock_train.return_value = TrainingResult(
                job_id="test",
                status=JobStatus.COMPLETED,
                work_dir=tmp_path / "manual_training",
                duration_seconds=1.0,
                log_content="done",
                potential_path=tmp_path / "manual_training" / "potential.yace",
                validation_metrics={"rmse_energy": 0.01}
            )

            # Mock select_active_set to just return dataset path
            with patch("mlip_autopipec.physics.training.pacemaker.PacemakerRunner.select_active_set") as mock_select:
                 mock_select.return_value = dataset_path

                 result = runner.invoke(app, ["train", str(dataset_path)])

                 assert result.exit_code == 0
                 assert "Training Completed Successfully" in result.stdout
                 assert "potential.yace" in result.stdout
