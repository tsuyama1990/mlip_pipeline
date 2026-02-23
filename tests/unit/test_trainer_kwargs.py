"""Test that Trainer passes kwargs correctly to underlying managers/wrappers."""

from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock, NonCallableMagicMock, patch

import pytest

from pyacemaker.core.config import (
    DistillationConfig,
    Step3MaceFinetuneConfig,
    TrainerConfig,
)
from pyacemaker.domain_models.models import (
    PotentialType,
    StructureMetadata,
    StructureStatus,
)
from pyacemaker.modules.trainer import MaceTrainer, PacemakerTrainer


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock config."""
    config = MagicMock()
    config.version = "1.0.0"
    config.trainer = TrainerConfig(
        potential_type="pace",
        mock=False,
        delta_learning="none"
    )
    config.distillation = DistillationConfig(
        step3_mace_finetune=Step3MaceFinetuneConfig(epochs=100)
    )
    config.oracle = MagicMock()
    config.oracle.mace = MagicMock()
    return config


def mock_dataset_gen() -> Iterator[StructureMetadata]:
    """Create a mock dataset generator."""
    from ase import Atoms
    real_atoms = Atoms("H2", positions=[[0,0,0], [0.74,0,0]])
    real_atoms.info = {}

    s = StructureMetadata(
        features={"atoms": real_atoms},
        energy=-10.0,
        forces=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], # Match atoms count
        status=StructureStatus.CALCULATED
    )
    yield s


def test_pacemaker_trainer_passes_kwargs(
    mock_config: MagicMock
) -> None:
    """Test PacemakerTrainer passes kwargs (e.g. weight_dft) to wrapper."""
    with patch("pyacemaker.modules.trainer.PacemakerWrapper") as MockWrapper, \
         patch("pyacemaker.modules.trainer.DatasetManager") as MockDM:

        wrapper_instance = MockWrapper.return_value
        wrapper_instance.train.return_value = Path("output.yace")

        # Make save_iter consume iterator so stats are updated
        def side_effect_save_iter(iterator: Any, *args: Any, **kwargs: Any) -> None:
            for _ in iterator:
                pass  # Consume without materializing full list

        MockDM.return_value.save_iter.side_effect = side_effect_save_iter

        expected_weight = 10.0
        expected_extra = "test"
        trainer = PacemakerTrainer(mock_config)

        # Call train with extra kwarg
        trainer.train(mock_dataset_gen(), weight_dft=expected_weight, extra_param=expected_extra)

        # Verify wrapper.train called with correct params
        args, kwargs = wrapper_instance.train.call_args
        params = args[2] # 3rd arg is params

        assert "weight_dft" in params
        assert params["weight_dft"] == expected_weight
        assert "extra_param" in params
        assert params["extra_param"] == expected_extra


def test_mace_trainer_passes_kwargs_and_epochs(
    mock_config: MagicMock
) -> None:
    """Test MaceTrainer passes kwargs and maps epochs correctly."""
    with patch("pyacemaker.trainer.mace_trainer.MaceManager") as MockManager, \
         patch("pyacemaker.trainer.mace_trainer.DatasetManager"):

        manager_instance = MockManager.return_value
        manager_instance.train.return_value = Path("output.model")

        trainer = MaceTrainer(mock_config)

        # Call train
        trainer.train(mock_dataset_gen(), foundation_model="base.model", extra_mace_param=True)

        # Verify manager.train called with correct params
        args, kwargs = manager_instance.train.call_args
        params = args[2] # 3rd arg is params

        expected_epochs = 100

        # Check epochs mapping (config has 100)
        assert "max_num_epochs" in params
        assert params["max_num_epochs"] == expected_epochs
        assert "epochs" not in params # Should not use "epochs" key

        # Check extra kwargs
        assert "extra_mace_param" in params
        assert params["extra_mace_param"] is True

        # Check foundation model logic from args (not kwargs)


def test_pacemaker_trainer_empty_dataset(
    mock_config: MagicMock
) -> None:
    """Test PacemakerTrainer handles empty dataset correctly."""
    expected_error = "No valid structures"
    with patch("pyacemaker.modules.trainer.PacemakerWrapper"), \
         patch("pyacemaker.modules.trainer.DatasetManager") as MockDM:

        # side effect to simulate empty iteration
        MockDM.return_value.save_iter.side_effect = lambda it, *a, **k: list(it)

        trainer = PacemakerTrainer(mock_config)

        with pytest.raises(ValueError, match=expected_error):
            trainer.train([])


def test_mace_trainer_empty_dataset(
    mock_config: MagicMock
) -> None:
    """Test MaceTrainer handles empty dataset without crashing (save_iter handles it)."""
    with patch("pyacemaker.trainer.mace_trainer.MaceManager") as MockManager, \
         patch("pyacemaker.trainer.mace_trainer.DatasetManager"):

        manager_instance = MockManager.return_value
        manager_instance.train.return_value = Path("output.model")

        trainer = MaceTrainer(mock_config)

        # Should just run and pass empty file to MACE (mocked)
        result = trainer.train([])

        assert result.type == PotentialType.MACE


def test_pacemaker_trainer_select_active_set(
    mock_config: MagicMock
) -> None:
    """Test PacemakerTrainer select_active_set."""
    with patch("pyacemaker.modules.trainer.PacemakerWrapper") as MockWrapper, \
         patch("pyacemaker.modules.trainer.DatasetManager") as MockDM:

        wrapper = MockWrapper.return_value
        wrapper.select_active_set.return_value = Path("selected.pckl.gzip")

        # Need consistent IDs
        candidates = list(mock_dataset_gen())

        # Mock load_iter to return atoms with uuid matching candidates
        from ase import Atoms
        mock_atoms = Atoms("H")
        mock_atoms.info = {"uuid": str(candidates[0].id)}
        MockDM.return_value.load_iter.return_value = iter([mock_atoms])

        n_select = 5
        trainer = PacemakerTrainer(mock_config)

        # We pass iterator, but since we materialized a list for ID checking, we pass iter(list)
        active_set = trainer.select_active_set(iter(candidates), n_select=n_select)

        assert str(candidates[0].id) in [str(u) for u in active_set.structure_ids]


def test_mace_trainer_select_active_set_raises(
    mock_config: MagicMock
) -> None:
    """Test MaceTrainer select_active_set raises NotImplementedError."""
    n_select = 5
    with patch("pyacemaker.trainer.mace_trainer.MaceManager"):
        trainer = MaceTrainer(mock_config)

        with pytest.raises(NotImplementedError):
            trainer.select_active_set(mock_dataset_gen(), n_select=n_select)
