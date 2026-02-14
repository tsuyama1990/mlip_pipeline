"""Tests for streaming safety in Trainer."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from ase import Atoms

from pyacemaker.core.config import PYACEMAKERConfig, TrainerConfig
from pyacemaker.domain_models.models import StructureMetadata
from pyacemaker.modules.trainer import PacemakerTrainer


@pytest.fixture
def trainer(tmp_path: Path) -> PacemakerTrainer:
    config = MagicMock(spec=PYACEMAKERConfig)
    # Use real config model to satisfy Pydantic validation later
    config.trainer = TrainerConfig(mock=True, potential_type="pace")
    return PacemakerTrainer(config)


def test_train_streaming_behavior(trainer: PacemakerTrainer) -> None:
    """Test that train does not consume generator into memory before passing to save_iter."""

    # Create a generator that we can track consumption of
    consumed_count = 0

    def data_gen() -> Iterator[StructureMetadata]:
        nonlocal consumed_count
        for _i in range(10):
            consumed_count += 1
            yield StructureMetadata(
                features={"atoms": Atoms("H")},
                energy=-1.0,
                forces=[[0.0, 0.0, 0.0]]
            )

    # Mock dataset_manager.save_iter to verify iterator type and consume it
    def side_effect(data, path, mode="wb"):
        assert isinstance(data, Iterator)
        assert not isinstance(data, list)
        # Consume to trigger counting
        list(data)

    save_iter_mock = MagicMock(side_effect=side_effect)
    trainer.dataset_manager.save_iter = save_iter_mock

    # Mock wrapper
    trainer.wrapper = MagicMock()

    trainer.train(data_gen())

    # Verify consumed
    assert consumed_count == 10


def test_select_active_set_streaming(trainer: PacemakerTrainer) -> None:
    """Test that active set selection streams data."""
    # Mock mocks
    trainer.dataset_manager = MagicMock()
    trainer.active_set_selector = MagicMock()
    trainer.dataset_manager.load_iter.return_value = iter([])  # Empty for result load

    candidates = (StructureMetadata(features={"atoms": Atoms("H")}) for _ in range(5))

    trainer.select_active_set(candidates, n_select=2)

    # Check save_iter called with iterator
    assert trainer.dataset_manager.save_iter.called
    args, _ = trainer.dataset_manager.save_iter.call_args
    assert isinstance(args[0], Iterator)
