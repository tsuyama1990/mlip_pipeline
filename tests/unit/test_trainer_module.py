"""Tests for high-level Trainer module."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, NonCallableMagicMock, patch

import pytest

from pyacemaker.core.config import PYACEMAKERConfig, TrainerConfig
from pyacemaker.domain_models.models import Potential, StructureMetadata
from pyacemaker.modules.trainer import PacemakerTrainer as Trainer


class TestTrainerModule:
    @pytest.fixture
    def config(self) -> MagicMock:
        # Mock PYACEMAKERConfig
        mock_config = MagicMock(spec=PYACEMAKERConfig)
        mock_config.trainer = TrainerConfig(
            cutoff=5.0,
            order=3,
            basis_size=(15, 5),
            delta_learning="zbl",
            max_epochs=100,
            batch_size=32,
        )
        return mock_config

    @pytest.fixture
    def trainer(self, config: MagicMock) -> Iterator[Trainer]:
        with (
            patch("pyacemaker.modules.trainer.PacemakerWrapper") as MockWrapper,
            patch("pyacemaker.modules.trainer.ActiveSetSelector") as MockSelector,
            patch("pyacemaker.modules.trainer.DatasetManager") as MockDatasetManager,
        ):
            trainer = Trainer(config=config)
            # Attach mocks to trainer instance for easy access in tests
            trainer.mock_wrapper = MockWrapper.return_value  # type: ignore[attr-defined]
            trainer.mock_selector = MockSelector.return_value  # type: ignore[attr-defined]
            trainer.mock_dataset_manager = MockDatasetManager.return_value  # type: ignore[attr-defined]
            yield trainer

    def test_train_method(self, trainer: Trainer) -> None:
        """Test trainer.train method."""
        # Create a valid structure with atoms, energy, and forces
        mock_atoms = NonCallableMagicMock()
        mock_atoms.copy.return_value = mock_atoms

        structure = StructureMetadata(
            features={"atoms": mock_atoms}, energy=-10.0, forces=[[0.0, 0.0, 0.0]]
        )
        dataset = [structure]

        # Configure mock return values
        trainer.mock_wrapper.train.return_value = Path("output.yace")  # type: ignore[attr-defined]

        # Ensure save_iter consumes the iterator to trigger counting AND creates file
        def consume_iterator(data: Iterator[Any], path: Path) -> None:
            for _ in data:
                pass
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        trainer.mock_dataset_manager.save_iter.side_effect = consume_iterator  # type: ignore[attr-defined]

        result = trainer.train(dataset)

        assert isinstance(result, Potential)
        trainer.mock_wrapper.train.assert_called_once()  # type: ignore[attr-defined]
        trainer.mock_dataset_manager.save_iter.assert_called_once()  # type: ignore[attr-defined]

    def test_initialization(self, trainer: Trainer) -> None:
        """Test initialization of Trainer module."""
        assert isinstance(trainer.trainer_config, TrainerConfig)
        # Verify sub-modules are initialized
        assert trainer.wrapper is not None
        assert trainer.active_set_selector is not None
        assert trainer.dataset_manager is not None
