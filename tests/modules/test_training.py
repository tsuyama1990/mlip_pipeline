"""
Unit tests for the PacemakerTrainer class.
"""

import subprocess
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
from ase.build import bulk

from mlip_autopipec.config.models import TrainingConfig, TrainingRunMetrics
from mlip_autopipec.modules.training import (
    NoTrainingDataError,
    PacemakerTrainer,
    TrainingFailedError,
)


@pytest.fixture
def mock_training_config(tmp_path: Path) -> TrainingConfig:
    """Creates a mock TrainingConfig for testing."""
    template_path = tmp_path / "pacemaker.in.j2"
    template_path.write_text("test template")
    executable_path = tmp_path / "pacemaker"
    executable_path.touch()
    executable_path.chmod(0o755)
    return TrainingConfig(
        pacemaker_executable=executable_path,
        data_source_db=tmp_path / "test.db",
        template_file=template_path,
    )


def test_pacemaker_trainer_perform_training(mock_training_config: TrainingConfig, tmp_path: Path):
    """
    Tests that the PacemakerTrainer correctly prepares input files and executes
    the training process.
    """
    # Arrange
    # Data is now passed explicitly
    atoms1 = bulk("Si")
    atoms1.info["energy"] = -10.0
    atoms1.arrays["forces"] = np.array([[0.1, 0.1, 0.1]] * len(atoms1))

    atoms2 = bulk("Si")
    atoms2.info["energy"] = -10.1
    atoms2.arrays["forces"] = np.array([[0.05, 0.05, 0.05]] * len(atoms2))

    training_data = [atoms1, atoms2]

    trainer = PacemakerTrainer(training_config=mock_training_config)

    with (
        patch("subprocess.run") as mock_run,
        patch("shutil.which", return_value=True),
        patch("tempfile.TemporaryDirectory") as mock_tmpdir,
    ):
        mock_tmpdir.return_value.__enter__.return_value = tmp_path
        # Mock stdout to include RMSE values and potential file
        stdout_output = (
            "Start training...\n"
            "Final potential saved to: potential.yace\n"
            "RMSE forces: 0.123\n"
            "RMSE energy: 0.045 eV/atom\n"  # Assuming this format, will adjust implementation
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout_output, stderr="")
        (tmp_path / "potential.yace").touch()

        # Act
        result_path, metrics = trainer.perform_training(training_data, generation=1)

    # Assert
    assert result_path.name == "potential.yace"
    assert isinstance(metrics, TrainingRunMetrics)
    assert metrics.num_structures == 2

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


def test_pacemaker_trainer_executable_not_found(mock_training_config: TrainingConfig):
    """
    Tests that a FileNotFoundError is raised if the pacemaker executable
    is not found.
    """
    atoms = bulk("Si")
    atoms.info["energy"] = -10.0
    atoms.arrays["forces"] = np.array([[0.1, 0.1, 0.1]] * len(atoms))
    training_data = [atoms]

    trainer = PacemakerTrainer(training_config=mock_training_config)
    with patch("shutil.which", return_value=False), pytest.raises(TrainingFailedError):
        trainer.perform_training(training_data, generation=1)


def test_pacemaker_trainer_training_failed(mock_training_config: TrainingConfig):
    """
    Tests that a TrainingFailedError is raised if the subprocess fails.
    """
    atoms = bulk("Si")
    atoms.info["energy"] = -10.0
    atoms.arrays["forces"] = np.array([[0.1, 0.1, 0.1]] * len(atoms))
    training_data = [atoms]

    trainer = PacemakerTrainer(training_config=mock_training_config)
    with (
        patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd")),
        patch("shutil.which", return_value=True),
        pytest.raises(TrainingFailedError),
    ):
        trainer.perform_training(training_data, generation=1)


def test_pacemaker_trainer_no_data(mock_training_config: TrainingConfig):
    """
    Tests that a NoTrainingDataError is raised if the input data is empty.
    """
    trainer = PacemakerTrainer(training_config=mock_training_config)
    with pytest.raises(NoTrainingDataError):
        trainer.perform_training([], generation=1)
