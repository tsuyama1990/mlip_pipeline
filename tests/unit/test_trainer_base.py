"""Unit tests for BaseTrainer."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.domain_models.models import (
    ActiveSet,
    Potential,
    PotentialType,
    StructureMetadata,
)
from pyacemaker.trainer.base import BaseTrainer


class ConcreteTrainer(BaseTrainer):
    """Concrete implementation of BaseTrainer for testing."""

    def train(
        self,
        dataset: Iterable[StructureMetadata],
        initial_potential: Potential | None = None,
        **kwargs: Any,
    ) -> Potential:
        """Mock train implementation."""
        return Potential(
            path=Path("mock.pot"),
            type=PotentialType.MACE,
            version="1.0",
            metrics={},
            parameters={},
        )

    def select_active_set(
        self, candidates: Iterable[StructureMetadata], n_select: int
    ) -> ActiveSet:
        """Mock select_active_set implementation."""
        return ActiveSet(
            structure_ids=[],
            structures=[],
            dataset_path=None,
            selection_criteria="mock",
        )

    def run(self) -> Any:
        """Mock run implementation."""


def test_base_trainer_init(full_config: PYACEMAKERConfig):
    """Test BaseTrainer initialization."""
    trainer = ConcreteTrainer(full_config)
    assert trainer.config == full_config
    assert trainer.trainer_config == full_config.trainer


def test_base_trainer_train_interface(full_config: PYACEMAKERConfig):
    """Test BaseTrainer train interface."""
    trainer = ConcreteTrainer(full_config)
    dataset = []
    potential = trainer.train(dataset)
    assert isinstance(potential, Potential)
    assert potential.path == Path("mock.pot")


def test_base_trainer_select_active_set_interface(full_config: PYACEMAKERConfig):
    """Test BaseTrainer select_active_set interface."""
    trainer = ConcreteTrainer(full_config)
    candidates = []
    active_set = trainer.select_active_set(candidates, n_select=10)
    assert isinstance(active_set, ActiveSet)
