import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator

from mlip_autopipec.domain_models.datastructures import Potential, Structure

logger = logging.getLogger(__name__)


class BaseDynamics(ABC):
    """Abstract Base Class for Dynamics Engine."""

    @abstractmethod
    def simulate(self, potential: Potential, structure: Structure) -> Iterator[Structure]:
        """
        Runs a simulation using the potential.

        Args:
            potential: The potential to use.
            structure: The initial structure.

        Returns:
            An iterator of structures representing the simulation trajectory.
        """


class MockDynamics(BaseDynamics):
    """Mock implementation of Dynamics Engine."""

    def simulate(self, potential: Potential, structure: Structure) -> Iterator[Structure]:
        logger.info("MockDynamics: Simulating trajectory...")

        initial_atoms = structure.to_ase()

        # Generate frames
        # For mock, we generate 5 frames.

        for i in range(5):
            atoms = initial_atoms.copy()
            # Perturb positions slightly
            positions = atoms.get_positions()
            positions += 0.01 * i
            atoms.set_positions(positions)

            yield Structure(
                atoms=atoms,
                provenance="md_trajectory",
                label_status="unlabeled"
            )
