from pathlib import Path
from unittest.mock import MagicMock, patch
import yaml

import pytest

from mlip_autopipec.domain_models.training import TrainingConfig, TrainingResult
from mlip_autopipec.domain_models.config import PotentialConfig
from mlip_autopipec.physics.training.pacemaker import PacemakerRunner


@pytest.fixture
def training_config():
    return TrainingConfig(
        max_epochs=10,
        batch_size=50,
        ladder_step=[5, 1],
        kappa=0.5
    )


@pytest.fixture
def potential_config():
    return PotentialConfig(
        elements=["Ti", "O"],
        cutoff=5.0,
        seed=42
    )


def test_train_success(tmp_path, training_config, potential_config):
    runner = PacemakerRunner(work_dir=tmp_path)
    dataset_path = tmp_path / "train.pckl.gzip"
    dataset_path.touch()

    # Mock subprocess and log file reading
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        def side_effect(*args, **kwargs):
             # Simulate pace_train outputting to stdout (which is redirected to file)
             if "stdout" in kwargs and hasattr(kwargs["stdout"], "write"):
                 kwargs["stdout"].write("RMSE Energy: 0.005\nRMSE Force: 0.1\n")
             # Also create the potential file which pace_train would do
             (tmp_path / "output_potential.yace").touch()

        mock_run.side_effect = side_effect

        result = runner.train(
            dataset_path=dataset_path,
            train_config=training_config,
            potential_config=potential_config
        )

        assert isinstance(result, TrainingResult)
        assert result.status.name == "COMPLETED"
        assert result.validation_metrics["energy_rmse"] == 0.005
        assert result.validation_metrics["force_rmse"] == 0.1

        # Check input.yaml generation
        input_yaml = tmp_path / "input.yaml"
        assert input_yaml.exists()
        data = yaml.safe_load(input_yaml.read_text())
        assert data["cutoff"] == 5.0
        assert data["elements"] == ["Ti", "O"]
        assert data["potential"]["embeddings"]["ZBL"]["type"] == "ZBL"


def test_active_set_selection(tmp_path, training_config, potential_config):
    runner = PacemakerRunner(work_dir=tmp_path)
    dataset_path = tmp_path / "train.pckl.gzip"
    dataset_path.touch()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        # Mock output file
        expected_output = tmp_path / "train_active.pckl.gzip"
        expected_output.touch()

        new_dataset_path = runner.select_active_set(dataset_path)

        assert new_dataset_path == expected_output
        mock_run.assert_called()
        args = mock_run.call_args[0][0]
        assert "pace_activeset" in args[0] or "pace_activeset" in args
