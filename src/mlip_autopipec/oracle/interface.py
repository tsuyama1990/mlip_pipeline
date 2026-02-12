import logging
from abc import ABC, abstractmethod

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from mlip_autopipec.domain_models.datastructures import Dataset, Structure

logger = logging.getLogger(__name__)


class BaseOracle(ABC):
    """Abstract Base Class for Oracle (DFT Engine)."""

    @abstractmethod
    def compute(self, structures: list[Structure]) -> Dataset:
        """
        Calculates ground-truth properties (energy, forces, stress) for structures.

        Args:
            structures: List of structures to compute.

        Returns:
            A Dataset containing the labeled structures.
        """


class MockOracle(BaseOracle):
    """Mock implementation of Oracle."""

    def compute(self, structures: list[Structure]) -> Dataset:
        logger.info("MockOracle: Computing energies and forces...")

        computed_structures = []
        for s in structures:
            # We copy structure to avoid mutating original list in-place if needed
            # But Structure wraps Atoms, which is mutable.
            # We will create a new Structure.

            # Create fake results
            atoms = s.atoms.copy()
            n_atoms = len(atoms)
            energy = -4.0 * n_atoms + np.random.normal(0, 0.1)
            forces = np.random.normal(0, 0.1, (n_atoms, 3))
            stress = np.random.normal(0, 0.01, 6)

            # Attach calculator results to ASE atoms
            calc = SinglePointCalculator(  # type: ignore[no-untyped-call]
                atoms,
                energy=energy,
                forces=forces,
                stress=stress
            )
            atoms.calc = calc

            # Create new Structure
            new_s = Structure(
                atoms=atoms,
                provenance=s.provenance,
                label_status="labeled",
                energy=energy,
                forces=forces.tolist(),
                stress=stress.tolist()
            )
            computed_structures.append(new_s)

        return Dataset(
            structures=computed_structures,
            description="Mock computed dataset"
        )
