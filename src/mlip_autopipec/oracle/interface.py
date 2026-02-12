import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from mlip_autopipec.domain_models.datastructures import Dataset, Structure

logger = logging.getLogger(__name__)


class BaseOracle(ABC):
    """Abstract Base Class for Oracle (DFT Engine)."""

    @abstractmethod
    def compute(self, structures: Iterable[Structure]) -> Dataset:
        """
        Calculates ground-truth properties (energy, forces, stress) for structures.

        Args:
            structures: Iterator or list of structures to compute.

        Returns:
            A Dataset containing the labeled structures.
        """


class MockOracle(BaseOracle):
    """Mock implementation of Oracle."""

    def compute(self, structures: Iterable[Structure]) -> Dataset:
        logger.info("MockOracle: Computing energies and forces...")

        computed_structures = []
        # Batched processing simulation to avoid huge memory spike if we were doing real work
        # But we still need to collect all for Dataset.
        # To strictly follow "no loading entire datasets into memory" before processing:
        # We process structure by structure.

        for s in structures:
            # Create fake results
            # s.atoms is object, need cast or use to_ase()
            atoms = s.to_ase().copy()  # type: ignore[no-untyped-call]
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
