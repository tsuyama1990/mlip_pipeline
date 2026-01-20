import gzip
import pickle
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.training import TrainConfig
from mlip_autopipec.training.pacemaker import PacemakerWrapper


@pytest.fixture
def mock_dataset_builder():
    builder = Mock()
    builder.export.return_value = Path("dummy_data.pckl.gzip")
    return builder


@pytest.fixture
def mock_config_gen():
    gen = Mock()
    gen.generate.return_value = Path("input.yaml")
    return gen


def test_pacemaker_train_success(mock_dataset_builder, mock_config_gen, tmp_path):
    # Setup dummy data file because wrapper reads it to get elements
    data_file = tmp_path / "dummy_data.pckl.gzip"
    atoms = [Atoms("AlCu")]
    with gzip.open(data_file, "wb") as f:
        pickle.dump(atoms, f)

    mock_dataset_builder.export.return_value = data_file

    wrapper = PacemakerWrapper(executable="pacemaker")
    config = TrainConfig()

    # Mock subprocess
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = """
        Training finished.
        RMSE (energy) : 0.005
        RMSE (forces) : 0.123
        """
        mock_run.return_value.returncode = 0

        # Create dummy output file
        (tmp_path / "potential_gen0.yace").touch()

        result = wrapper.train(
            config, mock_dataset_builder, mock_config_gen, tmp_path, generation=0
        )

        assert result.rmse_energy == 0.005
        assert result.rmse_forces == 0.123
        assert result.generation == 0
        assert result.potential_path == tmp_path / "potential_gen0.yace"

        # Verify subprocess call
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "pacemaker"
        assert args[1] == "input.yaml"


def test_pacemaker_train_failure(mock_dataset_builder, mock_config_gen, tmp_path):
    # Setup dummy data file
    data_file = tmp_path / "dummy_data.pckl.gzip"
    atoms = [Atoms("Al")]
    with gzip.open(data_file, "wb") as f:
        pickle.dump(atoms, f)
    mock_dataset_builder.export.return_value = data_file

    wrapper = PacemakerWrapper()
    config = TrainConfig()

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["pacemaker"], stderr="Error details"
        )

        with pytest.raises(RuntimeError, match="Pacemaker training failed"):
            wrapper.train(config, mock_dataset_builder, mock_config_gen, tmp_path)


def test_pacemaker_executable_not_found(mock_dataset_builder, mock_config_gen, tmp_path):
    # Setup dummy data file
    data_file = tmp_path / "dummy_data.pckl.gzip"
    atoms = [Atoms("Al")]
    with gzip.open(data_file, "wb") as f:
        pickle.dump(atoms, f)
    mock_dataset_builder.export.return_value = data_file

    wrapper = PacemakerWrapper(executable="non_existent_exe")
    config = TrainConfig()

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError

        with pytest.raises(RuntimeError, match="not found"):
            wrapper.train(config, mock_dataset_builder, mock_config_gen, tmp_path)


def test_pacemaker_parsing_failure(mock_dataset_builder, mock_config_gen, tmp_path):
    """Test when regex fails to find RMSE metrics."""
    data_file = tmp_path / "dummy_data.pckl.gzip"
    atoms = [Atoms("Al")]
    with gzip.open(data_file, "wb") as f:
        pickle.dump(atoms, f)
    mock_dataset_builder.export.return_value = data_file

    wrapper = PacemakerWrapper()
    config = TrainConfig()

    with patch("subprocess.run") as mock_run:
        # Mock output without RMSE patterns
        mock_run.return_value.stdout = "Training crashed without standard error code?"
        mock_run.return_value.returncode = 0

        (tmp_path / "potential_gen0.yace").touch()

        with pytest.raises(RuntimeError, match="parsing failed"):
            wrapper.train(config, mock_dataset_builder, mock_config_gen, tmp_path)
