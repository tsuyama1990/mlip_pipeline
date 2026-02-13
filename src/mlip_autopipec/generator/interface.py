import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ase import Atoms

from mlip_autopipec.domain_models.datastructures import Structure

if TYPE_CHECKING:
    from mlip_autopipec.domain_models.config import GeneratorConfig

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """Abstract Base Class for Structure Generators."""

    @abstractmethod
    def explore(self, context: dict[str, Any]) -> Iterator[Structure]:
        """
        Generates candidate structures.

        Args:
            context: Dictionary containing exploration parameters (e.g. temperature).

        Returns:
            An iterator of Structure objects.
        """

    def generate_local_candidates(
        self, structure: Structure, count: int = 5
    ) -> Iterator[Structure]:
        """
        Generates local candidates around a seed structure (e.g., for active learning).
        Default implementation returns just the seed to support subclasses that don't implement this yet.

        Args:
            structure: The seed structure (e.g. from a halt).
            count: Number of candidates to generate.

        Returns:
            An iterator of local candidate structures.
        """
        logger.warning(
            "Using default generate_local_candidates (no-op). Subclasses should override."
        )
        yield structure


class MockGenerator(BaseGenerator):
    """Mock implementation of Structure Generator."""

    def __init__(self, config: "GeneratorConfig | None" = None) -> None:
        self.config = config

    def explore(self, context: dict[str, Any]) -> Iterator[Structure]:
        logger.info("MockGenerator: Generating random structures...")

        # Make number configurable, default to 2
        default_count = self.config.mock_count if self.config else 2
        count = context.get("count", default_count)

        if isinstance(count, int):
            for i in range(count):
                # Create a simple dimer
                atoms = Atoms("He2", positions=[[0, 0, 0], [0, 0, 1.5 + i * 0.1]])
                yield Structure(atoms=atoms, provenance="mock_generator", label_status="unlabeled")
        else:
            # Fallback if not int
            pass

    def generate_local_candidates(
        self, structure: Structure, count: int = 5
    ) -> Iterator[Structure]:
        logger.info(f"MockGenerator: Generating {count} local candidates...")
        original_atoms = structure.to_ase()
        import numpy as np

        for i in range(count):
            atoms = original_atoms.copy()  # type: ignore[no-untyped-call]
            # Simple random rattle
            positions = atoms.get_positions()
            positions += np.random.normal(0, 0.05, positions.shape)
            atoms.set_positions(positions)

            yield Structure(
                atoms=atoms, provenance=f"local_candidate_{i}", label_status="unlabeled"
            )
