"""Tests for PacemakerTrainer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from pyacemaker.core.config import PYACEMAKERConfig, TrainerConfig
from pyacemaker.domain_models.models import PotentialType, StructureMetadata
from pyacemaker.trainer.pacemaker import PacemakerTrainer


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock configuration."""
    config = MagicMock(spec=PYACEMAKERConfig)
    # We construct TrainerConfig properly or mock it fields
    # Since we use config.trainer, let's mock it
    config.trainer = MagicMock(spec=TrainerConfig)
    config.trainer.potential_type = "pace"
    config.trainer.mock = False
    config.trainer.cutoff = 6.0
    config.trainer.basis_size = (1, 1)
    config.trainer.max_epochs = 10
    config.trainer.batch_size = 5
    config.trainer.delta_learning = "none"

    # We need model_dump to work
    config.trainer.model_dump.return_value = {
        "cutoff": 6.0,
        "max_epochs": 10,
        "batch_size": 5,
    }

    config.project = MagicMock()
    config.project.root_dir = tmp_path / "mock_project"
    config.version = "1.0.0"

    return config


@pytest.fixture
def mock_wrapper():
    """Mock PacemakerWrapper."""
    with patch("pyacemaker.trainer.pacemaker.PacemakerWrapper") as mock:
        yield mock


def test_pacemaker_init(mock_config, mock_wrapper):
    """Test initialization."""
    trainer = PacemakerTrainer(mock_config)
    assert trainer.wrapper is not None
    mock_wrapper.assert_called_once()


def test_generate_input_yaml(mock_config, mock_wrapper, tmp_path):
    """Test input.yaml generation."""
    trainer = PacemakerTrainer(mock_config)

    config_dict = {
        "cutoff": 5.0,
        "basis_size": (2, 2),
        "max_epochs": 20,
        "batch_size": 10
    }

    dataset_path = tmp_path / "data.xyz"
    work_dir = tmp_path

    yaml_path = trainer._generate_input_yaml(config_dict, dataset_path, work_dir)

    assert yaml_path.exists()

    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    assert data["cutoff"] == 5.0
    assert data["potential"]["bonds"]["N"] == 2
    assert data["backend"]["batch_size"] == 10
    assert data["optimizer"]["max_epochs"] == 20
    assert data["data"]["filename"] == str(dataset_path)


def test_train_mock_mode(mock_config, mock_wrapper):
    """Test train in mock mode."""
    mock_config.trainer.mock = True
    trainer = PacemakerTrainer(mock_config)

    # Mock dataset
    from ase import Atoms
    atoms = Atoms("H")
    s1 = StructureMetadata(energy=-1.0, forces=[[0,0,0]], features={"atoms": atoms})
    dataset = [s1]

    # Mock dataset manager to avoid file IO
    trainer.dataset_manager = MagicMock()
    # Ensure save_iter consumes the stream
    trainer.dataset_manager.save_iter.side_effect = lambda stream, *args, **kwargs: list(stream)

    potential = trainer.train(dataset)

    assert potential.type == PotentialType.PACE
    # Ensure wrapper.train was NOT called
    trainer.wrapper.train.assert_not_called()
    trainer.wrapper.train_from_input.assert_not_called()


def test_train_real_mode(mock_config, mock_wrapper):
    """Test train in real mode (calls wrapper)."""
    mock_config.trainer.mock = False
    trainer = PacemakerTrainer(mock_config)

    # Mock dataset
    from ase import Atoms
    atoms = Atoms("H")
    s1 = StructureMetadata(energy=-1.0, forces=[[0,0,0]], features={"atoms": atoms})
    dataset = [s1]

    trainer.dataset_manager = MagicMock()
    # Ensure save_iter consumes the stream
    trainer.dataset_manager.save_iter.side_effect = lambda stream, *args, **kwargs: list(stream)

    # Mock wrapper return
    mock_wrapper_instance = trainer.wrapper
    mock_wrapper_instance.train_from_input.return_value = Path("mock_output.yace")

    with (
        patch("pyacemaker.trainer.pacemaker.shutil.copy2") as mock_copy,
        patch("pathlib.Path.exists", return_value=True),
    ):
        potential = trainer.train(dataset)

        assert potential.type == PotentialType.PACE
        mock_wrapper_instance.train_from_input.assert_called_once()
        mock_copy.assert_called_once()
