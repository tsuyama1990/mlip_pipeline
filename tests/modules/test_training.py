"""Unit and integration tests for the PacemakerTrainer module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase.build import bulk
from ase.db import connect

from mlip_autopipec.config.training import TrainingConfig
from mlip_autopipec.modules.training import (
    PacemakerTrainer,
    TrainingFailedError,
)


@pytest.fixture
def mock_training_config(tmp_path: Path) -> TrainingConfig:
    """Create a mock TrainingConfig object for testing."""
    db_path = tmp_path / "test.db"
    template_path = tmp_path / "pacemaker.in.j2"
    template_path.touch()

    executable_path = tmp_path / "pacemaker"
    executable_path.touch()
    executable_path.chmod(0o755)

    return TrainingConfig(
        pacemaker_executable=executable_path,
        data_source_db=db_path,
        template_file=template_path,
    )


@pytest.fixture
def mock_ase_db(mock_training_config: TrainingConfig) -> Path:
    """Create a mock ASE database with a few structures."""
    db_path = mock_training_config.data_source_db
    with connect(db_path) as db:
        for i in range(5):
            atoms = bulk("Si", "diamond", a=5.43 + i * 0.1)
            db.write(atoms, data={"energy": -4.0 * len(atoms), "forces": (-0.1 * atoms.get_positions()).tolist()})
    return db_path


def test_pacemaker_trainer_initialization(mock_training_config: TrainingConfig):
    """Test that the PacemakerTrainer can be initialized."""
    trainer = PacemakerTrainer(training_config=mock_training_config)
    assert trainer.config == mock_training_config


def test_read_data_from_db(
    mock_training_config: TrainingConfig, mock_ase_db: Path
):
    """Test that the trainer can correctly read data from the ASE database."""
    trainer = PacemakerTrainer(training_config=mock_training_config)
    atoms_list = trainer._read_data_from_db()
    assert len(atoms_list) == 5
    assert "energy" in atoms_list[0].info
    assert "forces" in atoms_list[0].arrays


@patch("jinja2.Template.render")
def test_prepare_pacemaker_input(
    mock_render: MagicMock,
    mock_training_config: TrainingConfig,
    tmp_path: Path,
):
    """Test the generation of Pacemaker input files."""
    mock_render.return_value = "dummy config content"
    trainer = PacemakerTrainer(training_config=mock_training_config)
    atoms = bulk("Si")
    working_dir = tmp_path
    trainer._prepare_pacemaker_input([atoms], working_dir)
    mock_render.assert_called_once()
    assert (working_dir / "training_data.xyz").exists()
    assert (working_dir / "pacemaker.in").exists()
    assert (working_dir / "pacemaker.in").read_text() == "dummy config content"


@patch("subprocess.run")
def test_execute_training_success(
    mock_subprocess_run: MagicMock,
    mock_training_config: TrainingConfig,
    tmp_path: Path,
):
    """Test the successful execution of the training process."""
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=["pacemaker"],
        returncode=0,
        stdout="Final potential saved to: potential.yace",
        stderr="",
    )
    (tmp_path / "potential.yace").touch()
    trainer = PacemakerTrainer(training_config=mock_training_config)
    potential_path = trainer._execute_training(tmp_path)
    mock_subprocess_run.assert_called_once()
    assert potential_path.name == "potential.yace"


@patch("subprocess.run")
def test_execute_training_failure(
    mock_subprocess_run: MagicMock,
    mock_training_config: TrainingConfig,
    tmp_path: Path,
):
    """Test that the trainer raises an exception when training fails."""
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="pacemaker", stderr="Training failed"
    )
    trainer = PacemakerTrainer(training_config=mock_training_config)
    with pytest.raises(TrainingFailedError):
        trainer._execute_training(tmp_path)


@patch("mlip_autopipec.modules.training.PacemakerTrainer._read_data_from_db")
@patch("mlip_autopipec.modules.training.PacemakerTrainer._prepare_pacemaker_input")
@patch("mlip_autopipec.modules.training.PacemakerTrainer._execute_training")
def test_train_end_to_end_mocked(
    mock_execute: MagicMock,
    mock_prepare: MagicMock,
    mock_read: MagicMock,
    mock_training_config: TrainingConfig,
    tmp_path: Path,
):
    """Test the main train method with all helpers mocked."""
    mock_read.return_value = [bulk("Si")]
    (tmp_path / "potential.yace").touch()
    mock_execute.return_value = tmp_path / "potential.yace"

    trainer = PacemakerTrainer(training_config=mock_training_config)
    result_path = trainer.train()

    mock_read.assert_called_once()
    mock_prepare.assert_called_once()
    mock_execute.assert_called_once()

    assert result_path.name == "potential.yace"
    assert result_path.exists()
