# ruff: noqa: D101, D102, D103, D107, PT019, PT023
"""Tests for the PacemakerTrainer module."""

import subprocess
import typing
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from ase import Atoms
from pytest_mock import MockerFixture

from mlip_autopipec.config_schemas import SystemConfig
from mlip_autopipec.data.database import DatabaseManager
from mlip_autopipec.modules.config_generator import PacemakerConfigGenerator
from mlip_autopipec.modules.trainer import (
    NoTrainingDataError,
    PacemakerTrainer,
    TrainingFailedError,
)


@pytest.fixture
def test_system_config(tmp_path: Path) -> SystemConfig:
    """Provide a default SystemConfig for testing."""
    dft_config = {"executable": {}, "input": {"pseudopotentials": {"Ni": "ni.upf"}}}
    config = SystemConfig(dft=dft_config, db_path=str(tmp_path / "test.db"))
    config.trainer.loss_weights.energy = 2.0
    config.trainer.ace_params.correlation_order = 4
    return config


@pytest.fixture
def mock_db_manager(mocker: MockerFixture) -> DatabaseManager:
    """Provide a mocked DatabaseManager."""
    mock = mocker.MagicMock(spec=DatabaseManager)
    mock.get_completed_calculations.return_value = [Atoms("Ni"), Atoms("Ni2")]
    return typing.cast(DatabaseManager, mock)


@pytest.fixture
def mock_config_generator(mocker: MockerFixture) -> PacemakerConfigGenerator:
    """Provide a mocked PacemakerConfigGenerator."""
    mock = mocker.MagicMock(spec=PacemakerConfigGenerator)
    mock.generate_config.return_value = Path("mock_config.yaml")
    return typing.cast(PacemakerConfigGenerator, mock)


def test_train_orchestration_success(
    test_system_config: SystemConfig,
    mock_db_manager: DatabaseManager,
    mock_config_generator: PacemakerConfigGenerator,
    mocker: MockerFixture,
) -> None:
    """Integration test for a successful training workflow."""
    mocker.patch("shutil.which", return_value="pacemaker_train")
    mock_subprocess = mocker.patch("subprocess.run")
    mock_subprocess.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="INFO: Final potential saved to: potential.yace",
        stderr="",
    )

    # We need to manually inject the mocked generator
    trainer = PacemakerTrainer(test_system_config)
    trainer.config_generator = mock_config_generator

    atoms_list = mock_db_manager.get_completed_calculations()
    result_path = trainer.train(atoms_list)

    assert isinstance(mock_db_manager.get_completed_calculations, MagicMock)
    mock_db_manager.get_completed_calculations.assert_called_once()
    assert isinstance(mock_config_generator.generate_config, MagicMock)
    mock_config_generator.generate_config.assert_called_once()
    mock_subprocess.assert_called_once()
    assert "pacemaker_train" in mock_subprocess.call_args[0][0]
    assert result_path.endswith("potential.yace")


def test_train_failure_on_malformed_output(
    test_system_config: SystemConfig,
    mock_db_manager: DatabaseManager,
    mocker: MockerFixture,
) -> None:
    """Test that TrainingFailedError is raised on malformed subprocess output."""
    mocker.patch("shutil.which", return_value="pacemaker_train")
    mock_subprocess = mocker.patch("subprocess.run")
    mock_subprocess.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="Something went wrong, but no error code.",
        stderr="",
    )

    trainer = PacemakerTrainer(test_system_config)
    atoms_list = mock_db_manager.get_completed_calculations()
    with pytest.raises(
        TrainingFailedError, match="Could not find the output potential file"
    ):
        trainer.train(atoms_list)


def test_train_failure_on_subprocess_error(
    test_system_config: SystemConfig,
    mock_db_manager: DatabaseManager,
    mocker: MockerFixture,
) -> None:
    """Test that TrainingFailedError is raised when the subprocess fails."""
    mocker.patch("shutil.which", return_value="pacemaker_train")
    mock_subprocess = mocker.patch("subprocess.run")
    mock_subprocess.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd=[], stderr="Training crashed."
    )

    trainer = PacemakerTrainer(test_system_config)
    atoms_list = mock_db_manager.get_completed_calculations()
    with pytest.raises(TrainingFailedError, match="Training crashed."):
        trainer.train(atoms_list)


def test_train_failure_if_executable_not_found(
    test_system_config: SystemConfig,
    mock_db_manager: DatabaseManager,
    mocker: MockerFixture,
) -> None:
    """Test that FileNotFoundError is raised if the executable is not in PATH."""
    mocker.patch("shutil.which", return_value=None)
    trainer = PacemakerTrainer(test_system_config)
    atoms_list = mock_db_manager.get_completed_calculations()
    with pytest.raises(
        FileNotFoundError, match="Executable 'pacemaker_train' not found"
    ):
        trainer.train(atoms_list)


def test_no_training_data_raises_error(
    test_system_config: SystemConfig,
    mock_db_manager: DatabaseManager,
) -> None:
    """Test that NoTrainingDataError is raised when the database is empty."""
    assert isinstance(mock_db_manager.get_completed_calculations, MagicMock)
    mock_db_manager.get_completed_calculations.return_value = []

    trainer = PacemakerTrainer(test_system_config)
    with pytest.raises(NoTrainingDataError):
        trainer.train([])
