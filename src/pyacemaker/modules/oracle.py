"""Oracle (DFT) module implementation."""

import contextlib
import random

from ase import Atoms

from pyacemaker.core.base import ModuleResult
from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.exceptions import PYACEMAKERError
from pyacemaker.core.interfaces import Oracle
from pyacemaker.domain_models.models import StructureMetadata, StructureStatus
from pyacemaker.oracle.manager import DFTManager


class BaseOracle(Oracle):
    """Base implementation for Oracle modules with common utilities."""

    def _extract_atoms(self, structure: StructureMetadata) -> Atoms | None:
        """Extract ASE Atoms object from structure metadata."""
        atoms = structure.features.get("atoms")
        if isinstance(atoms, Atoms):
            return atoms
        self.logger.warning(f"Structure {structure.id} has no valid 'atoms' feature.")
        return None

    def _update_structure_common(
        self, structure: StructureMetadata, result_atoms: Atoms | None
    ) -> None:
        """Update structure metadata with results (Energy, Forces, Stress)."""
        if result_atoms is None:
            structure.status = StructureStatus.FAILED
            return

        structure.status = StructureStatus.CALCULATED
        try:
            # We use type: ignore because ASE types are often missing
            structure.features["energy"] = result_atoms.get_potential_energy()  # type: ignore[no-untyped-call]
            structure.features["forces"] = result_atoms.get_forces().tolist()  # type: ignore[no-untyped-call]

            # Stress might not always be calculated
            with contextlib.suppress(Exception):
                structure.features["stress"] = result_atoms.get_stress().tolist()  # type: ignore[no-untyped-call]

            structure.features["atoms"] = result_atoms
        except Exception:
            self.logger.exception(f"Failed to extract properties for {structure.id}")
            structure.status = StructureStatus.FAILED


class MockOracle(BaseOracle):
    """Mock Oracle implementation for testing."""

    def __init__(self, config: PYACEMAKERConfig) -> None:
        """Initialize the Mock Oracle."""
        super().__init__(config)
        # Use seeded random for determinism if configured, else strict determinism for tests
        self.seed = config.oracle.dft.parameters.get("seed", 42)
        self.rng = random.Random(self.seed)  # noqa: S311

    def run(self) -> ModuleResult:
        """Run the oracle (batch processing)."""
        self.logger.info("Running MockOracle")

        # Simulate failure based on config if needed
        if self.config.oracle.dft.parameters.get("simulate_failure", False):
            msg = "Simulated Oracle failure"
            raise PYACEMAKERError(msg)

        return ModuleResult(status="success")

    def compute_batch(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        """Compute energy/forces for a batch."""
        self.logger.info(f"Computing batch of {len(structures)} structures (mock)")

        for s in structures:
            # Update structure status
            s.status = StructureStatus.CALCULATED
            # Mock results with slight randomness using seeded RNG
            energy = -100.0 + self.rng.uniform(-1.0, 1.0)
            forces = [[self.rng.uniform(-0.1, 0.1) for _ in range(3)]]

            s.features["energy"] = energy
            s.features["forces"] = forces
            # No atoms update needed for mock simple

        return structures


class DFTOracle(BaseOracle):
    """Real DFT Oracle implementation using DFTManager."""

    def __init__(self, config: PYACEMAKERConfig) -> None:
        """Initialize the DFT Oracle."""
        super().__init__(config)
        self.dft_manager = DFTManager(config.oracle.dft)

    def run(self) -> ModuleResult:
        """Run the oracle (batch processing)."""
        # This method is from BaseModule, typically for standalone execution
        self.logger.info("Running DFTOracle")
        return ModuleResult(status="success")

    def compute_batch(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        """Compute energy/forces for a batch of structures.

        Args:
            structures: List of structure metadata to process.

        Returns:
            List of updated structure metadata.

        """
        self.logger.info(f"Computing batch of {len(structures)} structures (DFT)")

        chunk_size = 100
        total = len(structures)

        for i in range(0, total, chunk_size):
            chunk_structures = structures[i : i + chunk_size]
            chunk_atoms = []
            chunk_indices = []

            # Prepare chunk
            for j, s in enumerate(chunk_structures):
                atoms = self._extract_atoms(s)
                if atoms:
                    chunk_atoms.append(atoms)
                    chunk_indices.append(j) # Relative index in chunk

            if not chunk_atoms:
                continue

            # Process chunk (returns iterator, we consume it immediately to update)
            results_iter = self.dft_manager.compute_batch(chunk_atoms)

            for idx, result_atoms in zip(chunk_indices, results_iter, strict=True):
                self._update_structure_common(chunk_structures[idx], result_atoms)

        return structures
