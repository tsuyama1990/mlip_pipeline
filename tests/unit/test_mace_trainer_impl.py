"""Unit tests for MaceTrainer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.domain_models.models import (
    Potential,
    StructureMetadata,
)
from pyacemaker.trainer.mace_trainer import MaceTrainer


@pytest.fixture
def mace_config_mock(full_config: PYACEMAKERConfig):
    """Ensure MACE config is present."""
    full_config.oracle.mace.model_path = "mock.model"
    return full_config


@patch("pyacemaker.trainer.mace_trainer.MaceManager")
@patch("pyacemaker.trainer.mace_trainer.DatasetManager")
def test_mace_trainer_init(
    mock_dataset_manager: MagicMock,
    mock_mace_manager_cls: MagicMock,
    mace_config_mock: PYACEMAKERConfig,
):
    """Test MaceTrainer initialization."""
    trainer = MaceTrainer(mace_config_mock)
    mock_mace_manager_cls.assert_called_once_with(mace_config_mock.oracle.mace)
    assert trainer.mace_manager is not None


@patch("pyacemaker.trainer.mace_trainer.MaceManager")
@patch("pyacemaker.trainer.mace_trainer.DatasetManager")
def test_mace_trainer_train(
    mock_dataset_manager_cls: MagicMock,
    mock_mace_manager_cls: MagicMock,
    mace_config_mock: PYACEMAKERConfig,
):
    """Test MaceTrainer train method."""
    mock_mace_manager = mock_mace_manager_cls.return_value
    mock_mace_manager.train.return_value = Path("fine_tuned.model")

    mock_dataset_manager = mock_dataset_manager_cls.return_value

    trainer = MaceTrainer(mace_config_mock)

    expected_epochs = 10
    expected_model_path = Path("fine_tuned.model")

    # Create dummy dataset
    dataset: list[StructureMetadata] = []

    potential = trainer.train(dataset, epochs=expected_epochs)

    assert isinstance(potential, Potential)
    assert potential.path == expected_model_path
    assert potential.parameters["max_num_epochs"] == expected_epochs

    mock_dataset_manager.save_iter.assert_called_once()
    mock_mace_manager.train.assert_called_once()


@patch("pyacemaker.trainer.mace_trainer.MaceManager")
def test_mace_trainer_select_active_set_not_implemented(
    mock_mace_manager_cls: MagicMock,
    mace_config_mock: PYACEMAKERConfig,
):
    """Test select_active_set raises NotImplementedError."""
    trainer = MaceTrainer(mace_config_mock)
    with pytest.raises(NotImplementedError):
        trainer.select_active_set([], 10)
