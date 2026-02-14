"""Oracle (DFT) module implementation."""

import concurrent.futures
import contextlib
import random
import secrets
from collections.abc import Iterable, Iterator
from itertools import islice

from ase import Atoms

from pyacemaker.core.base import ModuleResult
from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.exceptions import PYACEMAKERError
from pyacemaker.core.interfaces import Oracle
from pyacemaker.core.utils import validate_structure_integrity
from pyacemaker.domain_models.models import StructureMetadata, StructureStatus
from pyacemaker.oracle.manager import DFTManager


class BaseOracle(Oracle):
    """Base implementation for Oracle modules with common utilities."""

    def validate_structure(self, structure: StructureMetadata) -> None:
        """Validate structure metadata before processing."""
        if not isinstance(structure, StructureMetadata):
            msg = f"Expected StructureMetadata, got {type(structure).__name__}"
            raise TypeError(msg)
        validate_structure_integrity(structure)

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
        return self.rng.uniform(a, b)

    def compute_batch(self, structures: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
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

            # Ensure atoms object exists for Trainer
            if "atoms" not in s.features or not isinstance(s.features.get("atoms"), Atoms):
                # Create dummy atoms
                s.features["atoms"] = Atoms(
                    "Fe", positions=[[0, 0, 0]], cell=[2.5, 2.5, 2.5], pbc=True
                )

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
        # Use a persistent executor to avoid overhead of creating one per chunk
        # Initialize lazily or in __init__? __init__ is better but BaseModule doesn't have close()
        # So we'll use a managed executor within compute_batch if possible, but compute_batch is a generator.
        # A generator cannot easily manage a "with executor" block that spans multiple yield calls if it's infinite.
        # However, typically compute_batch is finite.
        # But if we want true parallelism across chunks, we need the executor to live outside the chunk loop.
        # We will initialize it here but we must ensure it's used correctly.
        # Actually, Python's ThreadPoolExecutor is best used as a context manager.
        # Since compute_batch yields, we can wrap the whole loop in the executor context.

    def run(self) -> ModuleResult:
        """Run the oracle (batch processing)."""
        self.logger.info("Running DFTOracle")
        return ModuleResult(status="success")

    def compute_batch(self, structures: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
        """Compute energy/forces for a batch of structures."""
        self.logger.info("Computing batch of structures (DFT Parallel Streaming)")

        max_workers = self.config.oracle.dft.max_workers
        chunk_size = self.config.oracle.dft.chunk_size
        iterator = iter(structures)

        # We use a single ThreadPoolExecutor for the entire batch to avoid overhead
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            while True:
                # We still consume in chunks to avoid submitting infinite tasks if iterator is huge
                chunk = list(islice(iterator, chunk_size))
                if not chunk:
                    break

                # Submit chunk tasks
                # To maintain streaming behavior and not block on the whole chunk finishing,
                # we can submit all, yield futures as they complete.
                # However, to avoid OOM by submitting 1M tasks, we do it in chunks.
                # But we can process the chunk *using the shared executor*.
                yield from self._process_chunk_with_executor(chunk, executor)

    def _process_chunk_with_executor(
        self,
        chunk: list[StructureMetadata],
        executor: concurrent.futures.ThreadPoolExecutor,
    ) -> Iterator[StructureMetadata]:
        """Process a chunk using the provided executor."""
        to_process = []
        already_done = []

        for s in chunk:
            self.validate_structure(s)
            if s.status != StructureStatus.CALCULATED:
                atoms = self._extract_atoms(s)
                if atoms:
                    to_process.append((s, atoms))
                else:
                    s.status = StructureStatus.FAILED
                    already_done.append(s)
            else:
                already_done.append(s)

        yield from already_done

        if not to_process:
            return

        future_to_struct = {
            executor.submit(self.dft_manager.compute, atoms): s for s, atoms in to_process
        }

        for future in concurrent.futures.as_completed(future_to_struct):
            s = future_to_struct[future]
            try:
                result_atoms = future.result()
                self._update_structure_common(s, result_atoms)
            except Exception:
                self.logger.exception(f"Error computing structure {s.id}")
                s.status = StructureStatus.FAILED
            yield s
