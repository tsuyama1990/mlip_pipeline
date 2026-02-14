"""Core interfaces for PYACEMAKER modules."""

from abc import abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

from pyacemaker.core.base import BaseModule, ModuleResult
from pyacemaker.domain_models.models import (
    ActiveSet,
    CycleStatus,
    Potential,
    StructureMetadata,
)


class StructureGenerator(BaseModule):
    """Interface for structure generation module."""

    @abstractmethod
    def generate_initial_structures(self) -> list[StructureMetadata]:
        """Generate initial structures for cold start."""
        ...

    @abstractmethod
    def generate_local_candidates(
        self, seed_structure: StructureMetadata, n_candidates: int
    ) -> list[StructureMetadata]:
        """Generate candidate structures around a seed structure (e.g., high uncertainty)."""
        ...

    @abstractmethod
    def generate_batch_candidates(
        self, seed_structures: list[StructureMetadata], n_candidates_per_seed: int
    ) -> list[StructureMetadata]:
        """Generate candidate structures for a batch of seeds."""
        ...


class Oracle(BaseModule):
    """Interface for Oracle (DFT) module."""

    @abstractmethod
    def compute_batch(
        self, structures: Iterable[StructureMetadata]
    ) -> Iterator[StructureMetadata]:
        """Compute energy, forces, and stress for a batch of structures.

        Streaming interface: Takes an iterable and yields processed structures.
        """
        ...


class Trainer(BaseModule):
    """Interface for Trainer (Pacemaker) module."""

    @abstractmethod
    def train(
        self, dataset: list[StructureMetadata], initial_potential: Potential | None = None
    ) -> Potential:
        """Train a potential using the provided dataset."""
        ...

    @abstractmethod
    def select_active_set(self, candidates: list[StructureMetadata], n_select: int) -> ActiveSet:
        """Select the most informative structures from candidates."""
        ...


class DynamicsEngine(BaseModule):
    """Interface for Dynamics Engine (MD/kMC) module."""

    @abstractmethod
    def run_exploration(self, potential: Potential) -> list[StructureMetadata]:
        """Run MD exploration and return high-uncertainty structures."""
        ...

    @abstractmethod
    def run_production(self, potential: Potential) -> Any:
        """Run production MD/kMC simulation."""
        ...


class Validator(BaseModule):
    """Interface for Validator module."""

    @abstractmethod
    def validate(self, potential: Potential, test_set: list[StructureMetadata]) -> ModuleResult:
        """Validate the potential against a test set and physical constraints."""
        ...


class IOrchestrator(BaseModule):
    """Interface for the main Orchestrator."""

    @abstractmethod
    def run_cycle(self) -> CycleStatus:
        """Execute one active learning cycle."""
        ...
