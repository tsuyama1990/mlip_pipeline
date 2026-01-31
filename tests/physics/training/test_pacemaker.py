from unittest.mock import MagicMock, patch
import subprocess

import pytest

from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.training import TrainingConfig
from mlip_autopipec.domain_models.config import PotentialConfig
from mlip_autopipec.physics.training.pacemaker import PacemakerRunner


@pytest.fixture
def training_config():
    return TrainingConfig(
        batch_size=50,
        max_epochs=10,
        ladder_step=[20],
        kappa=0.5,
        active_set_optimization=False,
    )


@pytest.fixture
def potential_config():
    return PotentialConfig(
        elements=["Si", "O"],
        cutoff=5.0,
    )


@patch("subprocess.run")
def test_train_success(mock_run, training_config, potential_config, tmp_path):
    runner = PacemakerRunner(
        work_dir=tmp_path,
        train_config=training_config,
        potential_config=potential_config,
    )

    # Mock subprocess side effect to write to log file
    def mock_run_side_effect(cmd, **kwargs):
        if "stdout" in kwargs and hasattr(kwargs["stdout"], "write"):
            kwargs["stdout"].write("Final RMSE Energy: 0.005\nFinal RMSE Force: 0.01\n")
        return MagicMock(returncode=0)

    mock_run.side_effect = mock_run_side_effect

    dataset_path = tmp_path / "train.pckl.gzip"
    dataset_path.touch()

    # Create dummy potential file
    (tmp_path / "potential.yace").touch()

    result = runner.train(dataset_path)

    assert result.status == JobStatus.COMPLETED
    assert result.validation_metrics["energy"] == 0.005
    assert result.validation_metrics["force"] == 0.01
    assert result.potential_path == tmp_path / "potential.yace"

    # Check input.yaml created
    assert (tmp_path / "input.yaml").exists()

    # Check command
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "pace_train"
    assert any(str(arg).endswith("input.yaml") for arg in cmd)


@patch("subprocess.run")
def test_active_set_selection(mock_run, training_config, potential_config, tmp_path):
    runner = PacemakerRunner(
        work_dir=tmp_path,
        train_config=training_config,
        potential_config=potential_config,
    )

    dataset_path = tmp_path / "train.pckl.gzip"
    dataset_path.touch()

    mock_run.return_value.returncode = 0

    reduced_path = runner.select_active_set(dataset_path)

    assert reduced_path.name == "train_active.pckl.gzip"
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "pace_activeset"


@patch("subprocess.run")
def test_active_set_failure(mock_run, training_config, potential_config, tmp_path):
    # Enable active set
    training_config.active_set_optimization = True
    runner = PacemakerRunner(tmp_path, training_config, potential_config)
    dataset_path = tmp_path / "train.pckl.gzip"
    dataset_path.touch()

    # Mock active set failure
    def side_effect(cmd, **kwargs):
        if "pace_activeset" in cmd[0]:
            raise subprocess.CalledProcessError(1, cmd)
        # Mock pace_train success
        if "pace_train" in cmd[0]:
            # Create potential file
            (tmp_path / "potential.yace").touch()
            return MagicMock(returncode=0)
        return MagicMock(returncode=0)

    mock_run.side_effect = side_effect

    # It should catch error and proceed to train with full dataset
    result = runner.train(dataset_path)

    assert result.status == JobStatus.COMPLETED

    # Check calls
    # pace_activeset called, failed.
    # pace_train called.

    # Verify pace_train was called with input.yaml that points to original dataset
    input_yaml = tmp_path / "input.yaml"
    content = input_yaml.read_text()
    assert str(dataset_path) in content


@patch("subprocess.run")
def test_train_failure(mock_run, training_config, potential_config, tmp_path):
    runner = PacemakerRunner(tmp_path, training_config, potential_config)
    dataset_path = tmp_path / "train.pckl.gzip"
    dataset_path.touch()

    mock_run.side_effect = subprocess.CalledProcessError(1, "pace_train")

    result = runner.train(dataset_path)

    assert result.status == JobStatus.FAILED
