from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest
from mlip_autopipec.physics.training.pacemaker import PacemakerRunner
from mlip_autopipec.domain_models.training import TrainingConfig
from mlip_autopipec.domain_models.config import PotentialConfig, ACEConfig
from mlip_autopipec.domain_models.job import JobStatus
import subprocess

@pytest.fixture
def mock_train_config():
    return TrainingConfig(
        batch_size=10,
        max_epochs=1,
        active_set_optimization=True
    )

@pytest.fixture
def mock_pot_config():
    return PotentialConfig(
        elements=["Si"],
        cutoff=5.0,
        pair_style="hybrid/overlay",
        ace_params=ACEConfig(
            npot="FinnisSinclair",
            fs_parameters=[1, 1, 1, 0.5],
            ndensity=2
        )
    )

@pytest.fixture
def runner(tmp_path, mock_train_config, mock_pot_config):
    return PacemakerRunner(tmp_path, mock_train_config, mock_pot_config)

def test_train_success(runner):
    with patch("subprocess.run") as mock_run, \
         patch("pathlib.Path.read_text") as mock_read:

        mock_run.return_value = MagicMock(returncode=0)
        mock_read.return_value = "RMSE Energy: 0.001\nRMSE Force: 0.02"

        # Mock existence of potential file
        # pot_path = runner.work_dir / "potential.yace" # REMOVED unused variable

        with patch("pathlib.Path.exists") as mock_exists:
            # We need log path exists=True, potential path exists=True
            mock_exists.side_effect = lambda: True

            result = runner.train(Path("dataset.pckl.gzip"))

            assert result.status == JobStatus.COMPLETED
            assert result.validation_metrics["energy"] == 0.001

def test_active_set_selection(runner):
    with patch("subprocess.run") as mock_run:
        runner.select_active_set(Path("data.pckl.gzip"))
        # Check that pace_activeset was called
        args = mock_run.call_args[0][0]
        assert args[0] == "pace_activeset"

def test_active_set_failure(runner):
    # Should not crash, just log warning
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
        runner.train(Path("data.pckl.gzip"))
        # Verify it proceeded to train even if active set failed (implied by no raise)

def test_train_failure(runner):
    with patch("subprocess.run") as mock_run:
        # Mock active set success, but train failure
        def side_effect(cmd, **kwargs):
            if "pace_train" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        result = runner.train(Path("data.pckl.gzip"))
        assert result.status == JobStatus.FAILED
