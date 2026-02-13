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

        # Simulate batch processing
        batch_size = 50
        batch: list[Structure] = []

        # Max structures safety limit
        max_structures = 10000

        for count, s in enumerate(structures):
            if count >= max_structures:
                logger.warning(f"MockOracle reached max structure limit ({max_structures}). Stopping.")
                break

            batch.append(s)
            if len(batch) >= batch_size:
                yield from self._process_batch(batch, SinglePointCalculator, np)
                batch = []

        if batch:
            yield from self._process_batch(batch, SinglePointCalculator, np)

    def _process_batch(self, batch: list[Structure], calc_cls: Any, np_mod: Any) -> Iterator[Structure]:
        # Batch processing simulation
        for s in batch:
            atoms = s.to_ase().copy()  # type: ignore[no-untyped-call]
            n_atoms = len(atoms)
            energy = -4.0 * n_atoms + np_mod.random.normal(0, 0.1)
            forces = np_mod.random.normal(0, 0.1, (n_atoms, 3))
            stress = np_mod.random.normal(0, 0.01, 6)

            calc = calc_cls(
                atoms, energy=energy, forces=forces, stress=stress
            )
            atoms.calc = calc

            yield Structure(
                atoms=atoms,
                provenance=s.provenance,
                label_status="labeled",
                energy=energy,
                forces=forces.tolist(),
                stress=stress.tolist(),
            )
