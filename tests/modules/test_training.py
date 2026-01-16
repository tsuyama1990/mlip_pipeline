"""
Unit tests for the PacemakerTrainer class.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

import pytest
from ase.build import bulk
import subprocess

from mlip_autopipec.config.models import TrainingConfig
from mlip_autopipec.modules.training import PacemakerTrainer, TrainingFailedError, NoTrainingDataError


@pytest.fixture
def mock_training_config(tmp_path: Path) -> TrainingConfig:
    """Creates a mock TrainingConfig for testing."""
    template_path = tmp_path / "pacemaker.in.j2"
    template_path.write_text("test template")
    executable_path = tmp_path / "pacemaker"
    executable_path.touch()
    return TrainingConfig(
        pacemaker_executable=executable_path,
        data_source_db=tmp_path / "test.db",
        template_file=template_path,
    )


@patch("mlip_autopipec.modules.training.ase_db_connect")
def test_pacemaker_trainer_train(mock_db_connect, mock_training_config: TrainingConfig, tmp_path: Path):
    """
    Tests that the PacemakerTrainer correctly prepares input files and executes
    the training process.
    """
    # Arrange
    db_path = mock_training_config.data_source_db
    db_path.touch()
    mock_db_connect.return_value.__enter__.return_value.select.return_value = [
        MagicMock(toatoms=lambda: bulk("Si"), data={"energy": 0, "forces": [[0,0,0]]})
    ]
    trainer = PacemakerTrainer(training_config=mock_training_config)

    with patch("subprocess.run") as mock_run, \
         patch("shutil.which", return_value=True), \
         patch("tempfile.TemporaryDirectory") as mock_tmpdir:

        mock_tmpdir.return_value.__enter__.return_value = tmp_path
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Final potential saved to: potential.yace", stderr=""
        )
        (tmp_path / "potential.yace").touch()

        # Act
        result_path = trainer.train()

    # Assert
    assert result_path.name == "potential.yace"
    mock_run.assert_called_once_with(
        [str(mock_training_config.pacemaker_executable)],
        check=True,
        capture_output=True,
        text=True,
        cwd=ANY,
        shell=False,
    )
    # Check that the input file was created
    assert (tmp_path / "pacemaker.in").exists()
    assert (tmp_path / "training_data.xyz").exists()


@patch("mlip_autopipec.modules.training.ase_db_connect")
def test_pacemaker_trainer_executable_not_found(mock_db_connect, mock_training_config: TrainingConfig):
    """
    Tests that a FileNotFoundError is raised if the pacemaker executable
    is not found.
    """
    db_path = mock_training_config.data_source_db
    db_path.touch()
    mock_db_connect.return_value.__enter__.return_value.select.return_value = [
        MagicMock(toatoms=lambda: bulk("Si"), data={"energy": 0, "forces": [[0,0,0]]})
    ]
    trainer = PacemakerTrainer(training_config=mock_training_config)
    with patch("shutil.which", return_value=False):
        with pytest.raises(FileNotFoundError):
            trainer.train()


@patch("mlip_autopipec.modules.training.ase_db_connect")
def test_pacemaker_trainer_training_failed(mock_db_connect, mock_training_config: TrainingConfig):
    """
    Tests that a TrainingFailedError is raised if the subprocess fails.
    """
    db_path = mock_training_config.data_source_db
    db_path.touch()
    mock_db_connect.return_value.__enter__.return_value.select.return_value = [
        MagicMock(toatoms=lambda: bulk("Si"), data={"energy": 0, "forces": [[0,0,0]]})
    ]
    trainer = PacemakerTrainer(training_config=mock_training_config)
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd")), \
         patch("shutil.which", return_value=True):
        with pytest.raises(TrainingFailedError):
            trainer.train()


@patch("mlip_autopipec.modules.training.ase_db_connect")
def test_pacemaker_trainer_no_data(mock_db_connect, mock_training_config: TrainingConfig):
    """
    Tests that a NoTrainingDataError is raised if the database is empty.
    """
    db_path = mock_training_config.data_source_db
    db_path.touch()
    mock_db_connect.return_value.__enter__.return_value.select.return_value = []
    trainer = PacemakerTrainer(training_config=mock_training_config)
    with pytest.raises(NoTrainingDataError):
        trainer.train()
