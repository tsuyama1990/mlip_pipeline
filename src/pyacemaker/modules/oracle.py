"""Oracle (DFT) module implementation."""

import contextlib
import random
from collections.abc import Iterable, Iterator

from ase import Atoms

from pyacemaker.core.base import ModuleResult
from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.exceptions import PYACEMAKERError
from pyacemaker.core.interfaces import Oracle
from pyacemaker.domain_models.models import StructureMetadata, StructureStatus
from pyacemaker.oracle.manager import DFTManager


class BaseOracle(Oracle):
    """Base implementation for Oracle modules with common utilities."""

    def validate_structure(self, structure: StructureMetadata) -> None:
        """Validate structure metadata before processing."""
        if not isinstance(structure, StructureMetadata):
            msg = f"Expected StructureMetadata, got {type(structure).__name__}"
            raise TypeError(msg)

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
            # Update explicit fields
            structure.energy = float(result_atoms.get_potential_energy())  # type: ignore[no-untyped-call]
            structure.forces = result_atoms.get_forces().tolist()  # type: ignore[no-untyped-call]

            # Stress might not always be calculated
            with contextlib.suppress(Exception):
                structure.stress = result_atoms.get_stress().tolist()  # type: ignore[no-untyped-call]

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

    def compute_batch(
        self, structures: Iterable[StructureMetadata]
    ) -> Iterator[StructureMetadata]:
        """Compute energy/forces for a batch.

        Args:
            structures: Iterable of structure metadata.

        Yields:
            Updated structure metadata.

        """
        self.logger.info("Computing batch of structures (mock)")

        for s in structures:
            self.validate_structure(s)
            # Skip if already calculated
            if s.status == StructureStatus.CALCULATED:
                yield s
                continue

            # Update structure status
            s.status = StructureStatus.CALCULATED
            # Mock results with slight randomness using seeded RNG
            s.energy = -100.0 + self.rng.uniform(-1.0, 1.0)
            s.forces = [[self.rng.uniform(-0.1, 0.1) for _ in range(3)]]

            # No atoms update needed for mock simple
            yield s


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

    def _process_chunk(
        self,
        structures: list[StructureMetadata],
        atoms_list: list[Atoms],
        indices: list[int],
    ) -> None:
        """Helper to process a buffered chunk."""
        if not atoms_list:
            return

        results_iter = self.dft_manager.compute_batch(atoms_list)
        for idx, result_atoms in zip(indices, results_iter, strict=True):
            self._update_structure_common(structures[idx], result_atoms)

    def compute_batch(
        self, structures: Iterable[StructureMetadata]
    ) -> Iterator[StructureMetadata]:
        """Compute energy/forces for a batch of structures.

        Args:
            structures: Iterable of structure metadata to process.

        Yields:
            Updated structure metadata (streaming).

        """
        self.logger.info("Computing batch of structures (DFT)")

        chunk_size = self.config.oracle.dft.chunk_size

        current_chunk_structs: list[StructureMetadata] = []
        current_chunk_atoms: list[Atoms] = []
        current_chunk_indices: list[int] = []  # Relative index in chunk for valid atoms

        for s in structures:
            self.validate_structure(s)
            # 1. Validation check
            if s.status == StructureStatus.CALCULATED:
                # If we have a pending chunk, we must flush it to maintain order
                if current_chunk_structs:
                    self._process_chunk(
                        current_chunk_structs, current_chunk_atoms, current_chunk_indices
                    )
                    yield from current_chunk_structs
                    current_chunk_structs = []
                    current_chunk_atoms = []
                    current_chunk_indices = []

                yield s
                continue

            # 2. Extract atoms
            atoms = self._extract_atoms(s)

            # Add to chunk buffer
            current_chunk_structs.append(s)
            if atoms:
                current_chunk_atoms.append(atoms)
                current_chunk_indices.append(len(current_chunk_structs) - 1)

            # Process if chunk full
            if len(current_chunk_structs) >= chunk_size:
                self._process_chunk(
                    current_chunk_structs, current_chunk_atoms, current_chunk_indices
                )
                yield from current_chunk_structs

                # Reset buffers
                current_chunk_structs = []
                current_chunk_atoms = []
                current_chunk_indices = []

        # Process remaining
        if current_chunk_structs:
            self._process_chunk(
                current_chunk_structs, current_chunk_atoms, current_chunk_indices
            )
            yield from current_chunk_structs
