import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from ase import Atoms

from mlip_autopipec.domain_models.datastructures import Structure

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


class MockGenerator(BaseGenerator):
    """Mock implementation of Structure Generator."""

    def explore(self, context: dict[str, Any]) -> Iterator[Structure]:
        logger.info("MockGenerator: Generating random structures...")

        # Make number configurable, default to 2
        count = context.get("count", 2)
        if isinstance(count, int):
            for i in range(count):
                # Create a simple dimer
                atoms = Atoms("He2", positions=[[0, 0, 0], [0, 0, 1.5 + i * 0.1]])
                yield Structure(
                    atoms=atoms,
                    provenance="mock_generator",
                    label_status="unlabeled"
                )
        else:
            # Fallback if not int
            pass
