"""Test that Trainer passes kwargs correctly to underlying managers/wrappers."""

from pathlib import Path
from typing import Any
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
def mock_config(tmp_path: Path) -> MagicMock:
    """Create a mock config."""
    config = MagicMock()
    config.version = "1.0.0"
    config.project.root_dir = tmp_path
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


@pytest.fixture
def mock_dataset() -> list[StructureMetadata]:
    """Create a mock dataset."""
    mock_atoms = NonCallableMagicMock()
    # Mock atoms todict or copy behavior if needed
    mock_atoms.copy.return_value = mock_atoms
    mock_atoms.info = {}
    mock_atoms.arrays = {}
    from ase import Atoms
    real_atoms = Atoms("H2", positions=[[0,0,0], [0.74,0,0]])
    real_atoms.info = {}
    # real_atoms.arrays is already populated with positions/numbers

    s = StructureMetadata(
        features={"atoms": real_atoms},
        energy=-10.0,
        forces=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], # Match atoms count
        status=StructureStatus.CALCULATED
    )
    return [s]


def test_pacemaker_trainer_passes_kwargs(
    mock_config: MagicMock, mock_dataset: list[StructureMetadata], tmp_path: Path
) -> None:
    """Test PacemakerTrainer passes kwargs (e.g. weight_dft) to wrapper."""
    with patch("pyacemaker.trainer.pacemaker.PacemakerWrapper") as MockWrapper, \
         patch("pyacemaker.trainer.pacemaker.DatasetManager") as MockDM, \
         patch.object(PacemakerTrainer, "_generate_input_yaml") as MockGenYaml:

        wrapper_instance = MockWrapper.return_value
        # Use existing file for return value
        output_yace = tmp_path / "output.yace"
        output_yace.touch()
        wrapper_instance.train_from_input.return_value = output_yace
        MockGenYaml.return_value = Path("input.yaml")

        # Make save_iter consume iterator so stats are updated
        def side_effect_save_iter(iterator: Any, *args: Any, **kwargs: Any) -> None:
            for _ in iterator:
                pass  # Consume without materializing full list

        MockDM.return_value.save_iter.side_effect = side_effect_save_iter

        trainer = PacemakerTrainer(mock_config)

        # Call train with extra kwarg
        trainer.train(mock_dataset, weight_dft=10.0, extra_param="test")

        # Verify _generate_input_yaml called with correct params
        # Mock replaces method, so self is not passed unless autospec=True
        # args[0] is params dict
        args, _ = MockGenYaml.call_args
        params = args[0]

        # weight_dft is in params because we updated params with kwargs
        assert "weight_dft" in params
        assert params["weight_dft"] == 10.0
        assert "extra_param" in params
        assert params["extra_param"] == "test"


def test_mace_trainer_passes_kwargs_and_epochs(
    mock_config: MagicMock, mock_dataset: list[StructureMetadata], tmp_path: Path
) -> None:
    """Test MaceTrainer passes kwargs and maps epochs correctly."""
    with patch("pyacemaker.trainer.mace_trainer.MaceManager") as MockManager, \
         patch("pyacemaker.trainer.mace_trainer.DatasetManager"):

        manager_instance = MockManager.return_value
        output_path = tmp_path / "output.model"
        output_path.touch()
        manager_instance.train.return_value = output_path

        trainer = MaceTrainer(mock_config)

        # Call train
        trainer.train(mock_dataset, foundation_model="base.model", extra_mace_param=True)

        # Verify manager.train called with correct params
        args, kwargs = manager_instance.train.call_args
        params = args[2] # 3rd arg is params

        # Check epochs mapping (config has 100)
        assert "max_num_epochs" in params
        assert params["max_num_epochs"] == 100
        assert "epochs" not in params # Should not use "epochs" key

        # Check extra kwargs
        assert "extra_mace_param" in params
        assert params["extra_mace_param"] is True

        # Check foundation model logic from args (not kwargs)


def test_pacemaker_trainer_empty_dataset(
    mock_config: MagicMock
) -> None:
    """Test PacemakerTrainer handles empty dataset correctly."""
    with patch("pyacemaker.trainer.pacemaker.PacemakerWrapper"), \
         patch("pyacemaker.trainer.pacemaker.DatasetManager") as MockDM:

        # side effect to simulate empty iteration
        MockDM.return_value.save_iter.side_effect = lambda it, *a, **k: list(it)

        trainer = PacemakerTrainer(mock_config)

        with pytest.raises(ValueError, match="No valid structures"):
            trainer.train([])


def test_mace_trainer_empty_dataset(
    mock_config: MagicMock, tmp_path: Path
) -> None:
    """Test MaceTrainer handles empty dataset without crashing (save_iter handles it)."""
    with patch("pyacemaker.trainer.mace_trainer.MaceManager") as MockManager, \
         patch("pyacemaker.trainer.mace_trainer.DatasetManager"):

        manager_instance = MockManager.return_value
        output_path = tmp_path / "output.model"
        output_path.touch()
        manager_instance.train.return_value = output_path

        trainer = MaceTrainer(mock_config)

        # Should just run and pass empty file to MACE (mocked)
        result = trainer.train([])

        assert result.type == PotentialType.MACE


def test_pacemaker_trainer_select_active_set(
    mock_config: MagicMock, mock_dataset: list[StructureMetadata], tmp_path: Path
) -> None:
    """Test PacemakerTrainer select_active_set."""
    with patch("pyacemaker.trainer.pacemaker.PacemakerWrapper") as MockWrapper, \
         patch("pyacemaker.trainer.pacemaker.DatasetManager") as MockDM:

        wrapper = MockWrapper.return_value
        selected_path = tmp_path / "selected.pckl.gzip"
        selected_path.touch()
        wrapper.select_active_set.return_value = selected_path

        # Mock load_iter to return empty list or some atoms with uuid
        mock_atoms = NonCallableMagicMock()
        mock_atoms.info = {"uuid": str(mock_dataset[0].id)}
        MockDM.return_value.load_iter.return_value = iter([mock_atoms])

        trainer = PacemakerTrainer(mock_config)

        active_set = trainer.select_active_set(mock_dataset, n_select=5)

        assert str(mock_dataset[0].id) in [str(u) for u in active_set.structure_ids]


def test_mace_trainer_select_active_set_raises(
    mock_config: MagicMock, mock_dataset: list[StructureMetadata]
) -> None:
    """Test MaceTrainer select_active_set returns dummy set."""
    with patch("pyacemaker.trainer.mace_trainer.MaceManager"):
        trainer = MaceTrainer(mock_config)

        active_set = trainer.select_active_set(mock_dataset, n_select=5)
        assert len(active_set.structure_ids) == 0
        assert active_set.selection_criteria == "external_mace_al"
