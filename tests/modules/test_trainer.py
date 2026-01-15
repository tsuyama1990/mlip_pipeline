# ruff: noqa: D101, D102, D103, D107, PT019, PT023
"""Tests for the PacemakerTrainer module."""

import subprocess
from pathlib import Path

import pytest
import yaml
from ase import Atoms
from pytest_mock import MockerFixture

from mlip_autopipec.config_schemas import SystemConfig
from mlip_autopipec.data.database import DatabaseManager
from mlip_autopipec.modules.trainer import (
    NoTrainingDataError,
    PacemakerTrainer,
    TrainingFailedError,
)


@pytest.fixture
def test_system_config(tmp_path: Path) -> SystemConfig:
    """Provides a default SystemConfig for testing."""
    # The DFT config is not used by the trainer, but is required by the schema
    dft_config = {"executable": {}, "input": {"pseudopotentials": {"Ni": "ni.upf"}}}
    config = SystemConfig(dft=dft_config, db_path=str(tmp_path / "test.db"))
    config.trainer.loss_weights.energy = 2.0
    config.trainer.ace_params.correlation_order = 4
    return config


@pytest.fixture
def mock_db_manager(mocker: MockerFixture) -> DatabaseManager:
    """Provides a mocked DatabaseManager."""
    mock = mocker.Mock(spec=DatabaseManager)
    # Configure the mock to return a list of simple Atoms objects
    mock.get_completed_calculations.return_value = [Atoms("Ni"), Atoms("Ni2")]
    return mock


def test_generate_pacemaker_config(
    test_system_config: SystemConfig, mock_db_manager: DatabaseManager
):
    """Unit test for the Pacemaker config generation logic."""
    trainer = PacemakerTrainer(test_system_config, mock_db_manager)
    dummy_data_path = Path("/tmp/dummy_data.xyz")

    # Manually set the temp_dir to avoid creating it for this unit test
    trainer._temp_dir = dummy_data_path.parent

    config_path = trainer._generate_pacemaker_config(dummy_data_path)
    assert config_path.exists()

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    # Verify that the generated config matches the SystemConfig
    assert config_data["fit_params"]["dataset_filename"] == str(dummy_data_path)
    assert (
        config_data["fit_params"]["loss_weights"]["energy"]
        == test_system_config.trainer.loss_weights.energy
    )
    assert (
        config_data["fit_params"]["ace"]["correlation_order"]
        == test_system_config.trainer.ace_params.correlation_order
    )


def test_train_orchestration_success(
    test_system_config: SystemConfig,
    mock_db_manager: DatabaseManager,
    mocker: MockerFixture,
):
    """Integration test for a successful training workflow."""
    mock_subprocess = mocker.patch("subprocess.run")
    mock_subprocess.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="INFO: Final potential saved to: potential.yace",
        stderr="",
    )

    trainer = PacemakerTrainer(test_system_config, mock_db_manager)
    result_path = trainer.train()

    mock_db_manager.get_completed_calculations.assert_called_once()
    mock_subprocess.assert_called_once()
    # Check that the command includes "pacemaker_train"
    assert "pacemaker_train" in mock_subprocess.call_args[0][0]
    assert result_path.endswith("potential.yace")


def test_train_failure_on_subprocess_error(
    test_system_config: SystemConfig,
    mock_db_manager: DatabaseManager,
    mocker: MockerFixture,
):
    """Test that TrainingFailedError is raised when the subprocess fails."""
    mock_subprocess = mocker.patch("subprocess.run")
    mock_subprocess.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd=[], stderr="Training crashed."
    )

    trainer = PacemakerTrainer(test_system_config, mock_db_manager)
    with pytest.raises(TrainingFailedError, match="Training crashed."):
        trainer.train()


def test_no_training_data_raises_error(
    test_system_config: SystemConfig,
    mock_db_manager: DatabaseManager,
):
    """Test that NoTrainingDataError is raised when the database is empty."""
    # Configure the mock to return an empty list
    mock_db_manager.get_completed_calculations.return_value = []

    trainer = PacemakerTrainer(test_system_config, mock_db_manager)
    with pytest.raises(NoTrainingDataError):
        trainer.train()
