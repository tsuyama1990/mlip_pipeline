import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

from mlip_autopipec.domain_models.datastructures import Potential, Structure

if TYPE_CHECKING:
    from mlip_autopipec.domain_models.config import DynamicsConfig

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

    def __init__(self, config: "DynamicsConfig | None" = None) -> None:
        self.config = config

    def simulate(self, potential: Potential, structure: Structure) -> Iterator[Structure]:
        logger.info("MockDynamics: Simulating trajectory...")

        frame_count = 5
        if self.config:
            frame_count = self.config.mock_frames

        initial_atoms = structure.to_ase()

        for i in range(frame_count):
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
