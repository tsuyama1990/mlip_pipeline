import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from mlip_autopipec.domain_models.datastructures import Structure

logger = logging.getLogger(__name__)


class BaseOracle(ABC):
    """Abstract Base Class for Oracle (DFT Engine)."""

    @abstractmethod
    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        """
        Calculates ground-truth properties (energy, forces, stress) for structures.

        Args:
            structures: Iterator or list of structures to compute.

        Returns:
            An iterator of labeled structures.
        """


class MockOracle(BaseOracle):
    """Mock implementation of Oracle."""

    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        logger.info("MockOracle: Computing energies and forces...")

        # Batched processing simulation to avoid huge memory spike if we were doing real work
        # For mock, we can just yield one by one.

        import numpy as np
        from ase.calculators.singlepoint import SinglePointCalculator

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
                atoms, energy=energy, forces=forces, stress=stress
            )
            atoms.calc = calc

            # Create new Structure and yield
            yield Structure(
                atoms=atoms,
                provenance=s.provenance,
                label_status="labeled",
                energy=energy,
                forces=forces.tolist(),
                stress=stress.tolist(),
            )
