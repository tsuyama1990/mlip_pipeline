"""Oracle (DFT) module implementation."""

import contextlib
import random
import secrets
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

        # Prepare values before setting status to CALCULATED
        # to ensure atomic update and satisfy model validation if we were re-validating.
        # Note: Pydantic model validation runs on init/assignment, but here we update fields.
        try:
            # We use type: ignore because ASE types are often missing
            energy = float(result_atoms.get_potential_energy())  # type: ignore[no-untyped-call]
            forces = result_atoms.get_forces().tolist()  # type: ignore[no-untyped-call]
            stress = None

            with contextlib.suppress(Exception):
                stress = result_atoms.get_stress().tolist()  # type: ignore[no-untyped-call]

            # Update explicit fields
            structure.energy = energy
            structure.forces = forces
            if stress:
                structure.stress = stress

            structure.features["atoms"] = result_atoms
            # Set status last
            structure.status = StructureStatus.CALCULATED

        except Exception:
            self.logger.exception(f"Failed to extract properties for {structure.id}")
            structure.status = StructureStatus.FAILED


class MockOracle(BaseOracle):
    """Mock Oracle implementation for testing."""

    def __init__(self, config: PYACEMAKERConfig) -> None:
        """Initialize the Mock Oracle."""
        super().__init__(config)
        self.seed = config.oracle.dft.parameters.get("seed")
        # Use secrets for secure random if no seed provided, else random for determinism (tests)
        if self.seed is not None:
            self.rng = random.Random(self.seed)  # noqa: S311
        else:
            self.rng = secrets.SystemRandom()

    def run(self) -> ModuleResult:
        """Run the oracle (batch processing)."""
        self.logger.info("Running MockOracle")

        # Simulate failure based on config if needed
        if self.config.oracle.dft.parameters.get("simulate_failure", False):
            msg = "Simulated Oracle failure"
            raise PYACEMAKERError(msg)

        return ModuleResult(status="success")

    def _get_random_uniform(self, a: float, b: float) -> float:
        """Get random float using configured RNG."""
        if isinstance(self.rng, random.Random):
            return self.rng.uniform(a, b)
        # SystemRandom doesn't have uniform in all python versions? It does.
        return self.rng.uniform(a, b)

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

            # Generate random values
            energy = -100.0 + self._get_random_uniform(-1.0, 1.0)
            forces = [[self._get_random_uniform(-0.1, 0.1) for _ in range(3)]]

            # Update structure
            s.energy = energy
            s.forces = forces
            s.status = StructureStatus.CALCULATED
            yield s


class DFTOracle(BaseOracle):
    """Real DFT Oracle implementation using DFTManager."""

    def __init__(self, config: PYACEMAKERConfig) -> None:
        """Initialize the DFT Oracle."""
        super().__init__(config)
        self.dft_manager = DFTManager(config.oracle.dft)

    def run(self) -> ModuleResult:
        """Run the oracle (batch processing)."""
        self.logger.info("Running DFTOracle")
        return ModuleResult(status="success")

    def compute_batch(
        self, structures: Iterable[StructureMetadata]
    ) -> Iterator[StructureMetadata]:
        """Compute energy/forces for a batch of structures.

        Streaming implementation: Processes structures one-by-one to avoid memory overhead.

        Args:
            structures: Iterable of structure metadata to process.

        Yields:
            Updated structure metadata (streaming).

        """
        self.logger.info("Computing batch of structures (DFT Streaming)")

        for s in structures:
            self.validate_structure(s)

            # 1. Check status
            if s.status == StructureStatus.CALCULATED:
                yield s
                continue

            # 2. Extract atoms
            atoms = self._extract_atoms(s)
            if not atoms:
                # Failed to extract atoms, mark failed
                s.status = StructureStatus.FAILED
                yield s
                continue

            # 3. Compute (Delegate to DFTManager for single item)
            # DFTManager.compute returns the Atoms object with results attached
            result_atoms = self.dft_manager.compute(atoms)

            # 4. Update structure
            self._update_structure_common(s, result_atoms)
            yield s
