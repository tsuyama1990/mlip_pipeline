import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

from mlip_autopipec.domain_models.datastructures import Structure

if TYPE_CHECKING:
    from mlip_autopipec.domain_models.config import OracleConfig

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

    def __init__(self, config: "OracleConfig | None" = None) -> None:
        self.config = config

    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        logger.info("MockOracle: Computing energies and forces...")

        import numpy as np
        from ase.calculators.singlepoint import SinglePointCalculator

        # Max structures safety limit
        max_structures = 10000

        for count, s in enumerate(structures):
            if count >= max_structures:
                logger.warning(f"MockOracle reached max structure limit ({max_structures}). Stopping.")
                break

            # Process immediately (True Streaming) to guarantee O(1) memory
            # Batching logic removed as per audit request to process one-by-one or discard immediately
            yield self._process_single(s, SinglePointCalculator, np)

    def _process_single(self, structure: Structure, calc_cls: Any, np_mod: Any) -> Structure:
        atoms = structure.to_ase().copy()  # type: ignore[no-untyped-call]
        n_atoms = len(atoms)
        energy = -4.0 * n_atoms + np_mod.random.normal(0, 0.1)
        forces = np_mod.random.normal(0, 0.1, (n_atoms, 3))
        stress = np_mod.random.normal(0, 0.01, 6)

        calc = calc_cls(
            atoms, energy=energy, forces=forces, stress=stress
        )
        atoms.calc = calc

        return Structure(
            atoms=atoms,
            provenance=structure.provenance,
            label_status="labeled",
            energy=energy,
            forces=forces.tolist(),
            stress=stress.tolist(),
        )
