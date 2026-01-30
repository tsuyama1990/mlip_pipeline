from unittest.mock import patch

import pytest
import yaml

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.training import TrainingConfig
from mlip_autopipec.physics.training.pacemaker import PacemakerRunner


@pytest.fixture
def training_config():
    return TrainingConfig(
        batch_size=32,
        max_epochs=10,
        active_set_selection=True
    )


def test_train_generates_input_and_calls_pace_train(training_config, tmp_path):
    runner = PacemakerRunner(work_dir=tmp_path)
    dataset_path = tmp_path / "train.pckl.gzip"
    dataset_path.touch()

    with patch("subprocess.run") as mock_run:
        # Mock successful training
        mock_run.return_value.returncode = 0

        # Mock log file creation with RMSE
        (tmp_path / "log.txt").write_text("Energy RMSE: 1.23 meV/atom\nForce RMSE: 0.05 eV/A")

        # Mock potential file creation
        (tmp_path / "output_potential.yace").touch()

        potential = runner.train(training_config, dataset_path, elements=["Si"])

        assert isinstance(potential, Potential)
        assert potential.path == tmp_path / "output_potential.yace"
        assert potential.metadata["rmse_energy"] == 1.23
        assert potential.metadata["rmse_force"] == 0.05

        # Verify input.yaml generation
        input_yaml = tmp_path / "input.yaml"
        assert input_yaml.exists()
        with open(input_yaml) as f:
            data = yaml.safe_load(f)
            assert data["cutoff"] > 0
            assert data["data"]["filename"] == str(dataset_path)
            assert data["backend"]["batch_size"] == 32


def test_select_active_set_calls_pace_activeset(training_config, tmp_path):
    runner = PacemakerRunner(work_dir=tmp_path)
    dataset_path = tmp_path / "train.pckl.gzip"
    dataset_path.touch()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        # Mock output file
        expected_output = tmp_path / "train_active.pckl.gzip"
        expected_output.touch()

        result_path = runner.select_active_set(training_config, dataset_path)

        assert result_path == expected_output
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "pace_activeset" in args
