from unittest.mock import MagicMock, patch

from mlip_autopipec.domain_models.training import TrainingConfig
from mlip_autopipec.domain_models.config import PotentialConfig
from mlip_autopipec.physics.training.pacemaker import PacemakerRunner


def test_uat_full_training_loop(tmp_path):
    """
    Scenario 4.1: Full Training Loop
    """
    # Given a dataset
    dataset_path = tmp_path / "train.pckl.gzip"
    dataset_path.touch()

    config = TrainingConfig(max_epochs=10)
    pot_config = PotentialConfig(elements=["Ti", "O"], cutoff=5.0)

    runner = PacemakerRunner(work_dir=tmp_path)

    # Mock subprocess to simulate pace_train
    def side_effect(args, **kwargs):
        if args[0] == "pace_train":
            (runner.work_dir / "potential.yace").touch()
            (runner.work_dir / "log.txt").write_text(
                "RMSE Energy: 0.005\nRMSE Force: 0.1\n"
            )
            return MagicMock(returncode=0)
        return MagicMock(returncode=1)

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = side_effect

        # When I run training
        result = runner.train(dataset_path, config, pot_config)

        # Then a potential is created
        assert result.potential_path.exists()
        assert result.potential_path.name == "potential.yace"

        # And metrics are reported
        assert result.validation_metrics["rmse_energy"] == 0.005


def test_uat_active_set_selection(tmp_path):
    """
    Scenario 4.2: Active Set Selection
    """
    dataset_path = tmp_path / "train.pckl.gzip"
    dataset_path.touch()

    config = TrainingConfig(active_set_optimization=True)
    pot_config = PotentialConfig(elements=["Ti", "O"], cutoff=5.0)

    runner = PacemakerRunner(work_dir=tmp_path)

    def side_effect(args, **kwargs):
        if args[0] == "pace_activeset":
            (runner.work_dir / "train_active.pckl.gzip").touch()
            return MagicMock(returncode=0)
        return MagicMock(returncode=1)

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = side_effect

        active_path = runner.select_active_set(dataset_path, config, pot_config)

        assert active_path.name == "train_active.pckl.gzip"
        assert active_path.exists()
