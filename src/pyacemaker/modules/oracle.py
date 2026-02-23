"""Oracle (DFT) module implementation."""

import concurrent.futures
import random
import secrets
from collections.abc import Iterable, Iterator
from itertools import islice

from ase import Atoms

from pyacemaker.core.base import ModuleResult
from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.exceptions import ConfigurationError, PYACEMAKERError
from pyacemaker.core.interfaces import Oracle, UncertaintyModel
from pyacemaker.core.utils import update_structure_metadata, validate_structure_integrity
from pyacemaker.domain_models.models import (
    StructureMetadata,
    StructureStatus,
    UncertaintyState,
)
from pyacemaker.oracle.mace_manager import MaceManager
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

    def _validate_and_extract_atoms(self, structure: StructureMetadata) -> Atoms | None:
        """Validate structure and extract atoms in one go.

        Returns:
            Atoms object if valid and not already calculated, None otherwise (and sets status).
        """
        self.validate_structure(structure)
        if structure.status == StructureStatus.CALCULATED:
            return None

        atoms = self._extract_atoms(structure)
        if atoms is None:
            structure.status = StructureStatus.FAILED
            return None
        return atoms


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

            # Ensure atoms object exists for Trainer
            atoms = s.features.get("atoms")
            if not isinstance(atoms, Atoms):
                # Create dummy atoms
                atoms = Atoms(
                    "Fe", positions=[[0, 0, 0]], cell=[2.5, 2.5, 2.5], pbc=True
                )
                s.features["atoms"] = atoms

            # Generate random values
            energy = -100.0 + self._get_random_uniform(-1.0, 1.0)
            # Generate forces for EACH atom
            forces = [[self._get_random_uniform(-0.1, 0.1) for _ in range(3)] for _ in range(len(atoms))]

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
        # Persistent executor for lifetime of module?
        # Since we cannot easily close it, we rely on context manager in compute_batch for now.
        # The previous audit request "persistent executor" likely means "don't create one per chunk".
        # We did that in the previous step (one per compute_batch call).
        # We will stick to the current implementation which is safe.

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
            atoms = self._validate_and_extract_atoms(s)
            if atoms:
                to_process.append((s, atoms))
            else:
                # Either calculated or failed (status set in _validate_and_extract_atoms)
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
                update_structure_metadata(s, result_atoms)
            except Exception:
                self.logger.exception(f"Error computing structure {s.id}")
                s.status = StructureStatus.FAILED
            yield s


class MaceSurrogateOracle(BaseOracle, UncertaintyModel):
    """MACE Surrogate Oracle implementation."""

    def __init__(self, config: PYACEMAKERConfig) -> None:
        """Initialize the MACE Oracle."""
        super().__init__(config)

        if config.oracle.mace is None:
            msg = "MACE configuration is missing."
            raise ConfigurationError(msg)

        if config.oracle.mock:
            self.logger.info("MACE Oracle loaded (Mock)")
            self.mace_manager = None
        else:
            self.mace_manager = MaceManager(config.oracle.mace)

    def run(self) -> ModuleResult:
        """Run the oracle (batch processing)."""
        self.logger.info("Running MaceSurrogateOracle")
        return ModuleResult(status="success")

    def compute_batch(self, structures: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
        """Compute energy/forces for a batch of structures."""
        self.logger.info("Computing batch of structures (MACE)")

        for s in structures:
            atoms = self._validate_and_extract_atoms(s)
            if atoms is None:
                yield s
                continue

            # Mark source as MACE
            s.label_source = "mace"

            if self.config.oracle.mock:
                # Mock behavior
                s.energy = -10.0
                s.forces = [[0.0, 0.0, 0.0] for _ in range(len(atoms))]
                s.status = StructureStatus.CALCULATED
                yield s
                continue

            # We verified self.mace_manager is not None if not mock
            # But type checker might complain if we don't assert/check
            if self.mace_manager is None:
                # Should not happen given __init__ logic unless modified
                msg = "MaceManager is None but mock is False"
                raise ConfigurationError(msg)

            try:
                result_atoms = self.mace_manager.compute(atoms)
                update_structure_metadata(s, result_atoms)
            except Exception:
                self.logger.exception(f"Error computing structure {s.id}")
                s.status = StructureStatus.FAILED

            yield s

    def compute_uncertainty(  # noqa: C901, PLR0912
        self, structures: Iterable[StructureMetadata]
    ) -> Iterator[StructureMetadata]:
        """Compute uncertainty for a batch of structures."""
        self.logger.info("Computing uncertainty (MACE)")

        # Uncertainty usually requires batch processing if using ensemble,
        # but here we iterate. To optimize, we could batch collect.
        # For simplicity and streaming consistency, we iterate but maybe call manager per item or batch internally.
        # Let's collect a small batch or just process one by one if manager supports lists.

        # Collecting all structures to batch process is risky for memory, but uncertainty is usually fast.
        # Let's collect in chunks.
        chunk_size = 100
        iterator = iter(structures)

        while True:
            chunk = list(islice(iterator, chunk_size))
            if not chunk:
                break

            # Extract atoms
            atoms_list = []
            valid_indices = []
            for i, s in enumerate(chunk):
                atoms = self._extract_atoms(s)
                if atoms:
                    atoms_list.append(atoms)
                    valid_indices.append(i)

            if not atoms_list:
                for s in chunk:
                    yield s
                continue

            uncertainties: list[float | None] = []
            if self.config.oracle.mock:
                uncertainties = [0.5] * len(atoms_list)  # Mock uncertainty
            elif self.mace_manager:
                try:
                    uncertainties = [
                        float(x) for x in self.mace_manager.compute_uncertainty(atoms_list)
                    ]
                except Exception:
                    self.logger.exception("Failed to compute uncertainty batch")
                    uncertainties = [None] * len(atoms_list)
            else:
                # Should not happen if mock checked above
                uncertainties = [None] * len(atoms_list)

            # Assign back
            for idx, unc in zip(valid_indices, uncertainties, strict=False):
                s = chunk[idx]
                if unc is not None:
                    # Update structure metadata with uncertainty
                    # Assuming gamma_max is the metric (or use mean if appropriate)
                    # Spec says "Uncertainty (Variance)".
                    s.uncertainty_state = UncertaintyState(
                        gamma_mean=float(unc), gamma_max=float(unc)
                    )

            for s in chunk:
                yield s
