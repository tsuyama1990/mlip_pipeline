"""Oracle (DFT) module implementation."""

import random
from collections.abc import Iterator

from ase import Atoms

from pyacemaker.core.base import ModuleResult
from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.exceptions import PYACEMAKERError
from pyacemaker.core.interfaces import Oracle
from pyacemaker.domain_models.models import StructureMetadata, StructureStatus
from pyacemaker.oracle.manager import DFTManager


class MockOracle(Oracle):
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
        # Since oracle.dft.parameters is a dict, we can check it
        if self.config.oracle.dft.parameters.get("simulate_failure", False):
            msg = "Simulated Oracle failure"
            raise PYACEMAKERError(msg)

        return ModuleResult(status="success")

    def compute_batch(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        """Compute energy/forces for a batch."""
        self.logger.info(f"Computing batch of {len(structures)} structures (mock)")

        computed = []
        for s in structures:
            # Update structure status
            s.status = StructureStatus.CALCULATED
            # Mock results with slight randomness using seeded RNG
            energy = -100.0 + self.rng.uniform(-1.0, 1.0)
            forces = [[self.rng.uniform(-0.1, 0.1) for _ in range(3)]]

            s.features["energy"] = energy
            s.features["forces"] = forces
            computed.append(s)

        return computed


class DFTOracle(Oracle):
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

    def _extract_atoms(self, structure: StructureMetadata) -> Atoms | None:
        """Extract ASE Atoms object from structure metadata."""
        atoms = structure.features.get("atoms")
        if isinstance(atoms, Atoms):
            return atoms
        self.logger.warning(f"Structure {structure.id} has no valid 'atoms' feature.")
        return None

    def _update_structure(self, structure: StructureMetadata, result_atoms: Atoms | None) -> None:
        """Update structure metadata with DFT results."""
        if result_atoms is None:
            structure.status = StructureStatus.FAILED
            return

        structure.status = StructureStatus.CALCULATED
        try:
            # We use type: ignore because ASE types are often missing
            structure.features["energy"] = result_atoms.get_potential_energy()  # type: ignore[no-untyped-call]
            structure.features["forces"] = result_atoms.get_forces().tolist()  # type: ignore[no-untyped-call]
            structure.features["stress"] = result_atoms.get_stress().tolist()  # type: ignore[no-untyped-call]
            # Update atoms object in features (e.g. relaxed structure)
            structure.features["atoms"] = result_atoms
        except Exception:
            # Catch all exceptions during extraction (e.g., property missing)
            self.logger.exception(f"Failed to extract properties for {structure.id}")
            structure.status = StructureStatus.FAILED

    def compute_batch(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        """Compute energy/forces for a batch of structures.

        Args:
            structures: List of structure metadata to process.

        Returns:
            List of updated structure metadata.

        """
        self.logger.info(f"Computing batch of {len(structures)} structures (DFT)")

        # Prepare generator for atoms
        def atoms_generator() -> Iterator[Atoms]:
            for s in structures:
                atoms = self._extract_atoms(s)
                if atoms:
                    yield atoms
                else:
                    # Yield a dummy atom to maintain sequence or handle missing
                    # Actually DFTManager expects valid atoms.
                    # If we skip, the mapping breaks.
                    # We must handle this carefully.
                    # Strategy: Filter valid indices first.
                    pass

        # To handle mapping correctly with a generator pipeline, we might need a different approach.
        # But compute_batch in interface expects returning the full list.
        # So we process linearly but efficiently.

        valid_indices = []
        valid_atoms = []

        for i, s in enumerate(structures):
            atoms = self._extract_atoms(s)
            if atoms:
                valid_indices.append(i)
                valid_atoms.append(atoms)

        if not valid_atoms:
            self.logger.warning("No valid atoms found in batch.")
            return structures

        # Run DFT via manager (streaming)
        results_iter = self.dft_manager.compute_batch(valid_atoms)

        # Update metadata
        for idx, result_atoms in zip(valid_indices, results_iter, strict=True):
            self._update_structure(structures[idx], result_atoms)

        return structures
