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
            # Enforce a hard limit for mock safety
            frame_count = min(self.config.mock_frames, 10000)

        # Create a copy to ensure we don't modify the original structure's atoms
        # via any side effects in to_ase() if it were to change.
        # to_ase returns the internal atoms object with updated info.
        initial_atoms = structure.to_ase().copy()  # type: ignore[no-untyped-call]

        for i in range(frame_count):
            logger.debug(f"MockDynamics: Frame {i}/{frame_count}")
            # Stream frames (generator) - ensures O(1) memory usage if consumed properly
            atoms = initial_atoms.copy()  # type: ignore[no-untyped-call]
            # Perturb positions slightly
            positions = atoms.get_positions()
            positions += 0.01 * i
            atoms.set_positions(positions)

            # Simulate increasing uncertainty
            score = 1.0 + (i * 1.5)
            metadata = {"step": i, "temperature": 300.0}

            yield Structure(
                atoms=atoms,
                provenance="md_trajectory",
                label_status="unlabeled",
                uncertainty_score=score,
                metadata=metadata,
            )
