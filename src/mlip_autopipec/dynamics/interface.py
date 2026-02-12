import logging
from abc import ABC, abstractmethod

from mlip_autopipec.domain_models.datastructures import Potential, Structure, Trajectory

logger = logging.getLogger(__name__)


class BaseDynamics(ABC):
    """Abstract Base Class for Dynamics Engine."""

    @abstractmethod
    def simulate(self, potential: Potential, structure: Structure) -> Trajectory:
        """
        Runs a simulation using the potential.

        Args:
            potential: The potential to use.
            structure: The initial structure.

        Returns:
            A Trajectory object containing the simulation frames.
        """


class MockDynamics(BaseDynamics):
    """Mock implementation of Dynamics Engine."""

    def simulate(self, potential: Potential, structure: Structure) -> Trajectory:
        logger.info("MockDynamics: Simulating trajectory...")

        trajectory_structures = []
        initial_atoms = structure.to_ase()

        for i in range(5):
            atoms = initial_atoms.copy()  # type: ignore[no-untyped-call]
            # Perturb positions slightly
            positions = atoms.get_positions()
            positions += 0.01 * i
            atoms.set_positions(positions)

            s = Structure(
                atoms=atoms,
                provenance="md_trajectory",
                label_status="unlabeled"
            )
            trajectory_structures.append(s)

        return Trajectory(
            structures=trajectory_structures,
            metadata={"potential": str(potential.path), "steps": 5}
        )
