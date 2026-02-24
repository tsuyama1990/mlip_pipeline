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
    tmp_path: Path,
):
    """Test MaceTrainer train method."""
    mock_mace_manager = mock_mace_manager_cls.return_value

    # Mock return path as absolute path in tmp_path and create it
    output_model = tmp_path / "fine_tuned.model"
    output_model.touch()
    mock_mace_manager.train.return_value = output_model

    mock_dataset_manager = mock_dataset_manager_cls.return_value

    trainer = MaceTrainer(mace_config_mock)

    # Create dummy dataset
    dataset: list[StructureMetadata] = []

    potential = trainer.train(dataset, epochs=10)

    assert isinstance(potential, Potential)
    assert potential.path.name.startswith("mace_model_")
    assert potential.parameters["max_num_epochs"] == 10

    mock_dataset_manager.save_iter.assert_called_once()
    mock_mace_manager.train.assert_called_once()


@patch("pyacemaker.trainer.mace_trainer.MaceManager")
def test_mace_trainer_select_active_set_dummy(
    mock_mace_manager_cls: MagicMock,
    mace_config_mock: PYACEMAKERConfig,
):
    """Test select_active_set returns dummy ActiveSet."""
    trainer = MaceTrainer(mace_config_mock)
    active_set = trainer.select_active_set([], 10)
    assert len(active_set.structure_ids) == 0
    assert active_set.selection_criteria == "external_mace_al"
