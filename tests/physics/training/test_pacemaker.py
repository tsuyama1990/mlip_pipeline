from unittest.mock import MagicMock, patch

import pytest
from mlip_autopipec.domain_models.training import TrainingConfig, TrainingResult
from mlip_autopipec.domain_models.config import PotentialConfig
from mlip_autopipec.physics.training.pacemaker import PacemakerRunner


@pytest.fixture
def training_config():
    return TrainingConfig(batch_size=50, max_epochs=10)


@pytest.fixture
def potential_config():
    return PotentialConfig(elements=["Ti", "O"], cutoff=5.0)


def test_train_success(tmp_path, training_config, potential_config):
    runner = PacemakerRunner(work_dir=tmp_path)
    dataset_path = tmp_path / "train.pckl.gzip"
    dataset_path.touch()

    # Create a dummy log file that the runner would expect to read
    # But wait, the runner runs subprocess then reads log.
    # So I need the mock_run to create the log file? Or I create it before?
    # Usually the log is created BY the command.
    # So I will patch subprocess.run to create the log file as a side effect.

    def side_effect_pace_train(args, **kwargs):
        # Create output files
        (runner.work_dir / "potential.yace").touch()
        (runner.work_dir / "log.txt").write_text(
            "RMSE Energy: 0.005\nRMSE Force: 0.1\n"
        )
        return MagicMock(returncode=0)

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = side_effect_pace_train

        result = runner.train(dataset_path, training_config, potential_config)

        assert isinstance(result, TrainingResult)
        assert result.potential_path == runner.work_dir / "potential.yace"
        assert result.validation_metrics["rmse_energy"] == 0.005
        assert result.validation_metrics["rmse_force"] == 0.1

        # Verify input.yaml creation
        assert (runner.work_dir / "input.yaml").exists()


def test_select_active_set(tmp_path, training_config, potential_config):
    runner = PacemakerRunner(work_dir=tmp_path)
    dataset_path = tmp_path / "train.pckl.gzip"
    dataset_path.touch()

    expected_output = runner.work_dir / "train_active.pckl.gzip"

    def side_effect_activeset(args, **kwargs):
        expected_output.touch()
        return MagicMock(returncode=0)

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = side_effect_activeset

        output = runner.select_active_set(
            dataset_path, training_config, potential_config
        )

        assert output == expected_output
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "pace_activeset"
