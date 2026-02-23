"""Oracle (DFT) module implementation."""

import concurrent.futures
import random
import secrets
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Any

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
        if self.seed is not None:
            self.rng = random.Random(self.seed)  # noqa: S311
        else:
            self.rng = secrets.SystemRandom()

    def run(self) -> ModuleResult:
        """Run the oracle (batch processing)."""
        self.logger.info("Running MockOracle")

        if self.config.oracle.dft.parameters.get("simulate_failure", False):
            msg = "Simulated Oracle failure"
            raise PYACEMAKERError(msg)

        return ModuleResult(status="success")

    def _get_random_uniform(self, a: float, b: float) -> float:
        """Get random float using configured RNG."""
        return self.rng.uniform(a, b)

    def compute_batch(self, structures: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
        """Compute energy/forces for a batch."""
        self.logger.info("Computing batch of structures (mock)")

        for s in structures:
            self.validate_structure(s)
            if s.status == StructureStatus.CALCULATED:
                yield s
                continue

            atoms = s.features.get("atoms")
            if not isinstance(atoms, Atoms):
                atoms = Atoms(
                    "Fe", positions=[[0, 0, 0]], cell=[2.5, 2.5, 2.5], pbc=True
                )
                s.features["atoms"] = atoms

            energy = -100.0 + self._get_random_uniform(-1.0, 1.0)
            forces = [[self._get_random_uniform(-0.1, 0.1) for _ in range(3)] for _ in range(len(atoms))]

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

    def compute_batch(self, structures: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
        """Compute energy/forces for a batch of structures."""
        self.logger.info("Computing batch of structures (DFT Parallel Streaming)")

        max_workers = self.config.oracle.dft.max_workers
        buffer_size = max_workers * 2

        iterator = iter(structures)
        futures: dict[concurrent.futures.Future, StructureMetadata] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            while True:
                # Refill
                while len(futures) < buffer_size:
                    try:
                        s = next(iterator)
                    except StopIteration:
                        break

                    atoms = self._validate_and_extract_atoms(s)
                    if atoms:
                        future = executor.submit(self.dft_manager.compute, atoms)
                        futures[future] = s
                    else:
                        yield s

                if not futures:
                    break

                # Wait
                done, _ = concurrent.futures.wait(
                    futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                )

                for future in done:
                    s = futures.pop(future)
                    self._process_result(s, future)
                    yield s

    def _process_result(self, s: StructureMetadata, future: concurrent.futures.Future) -> None:
        try:
            result_atoms = future.result()
            update_structure_metadata(s, result_atoms)
        except Exception:
            self.logger.exception(f"Error computing structure {s.id}")
            s.status = StructureStatus.FAILED


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

        batch_size = self.config.oracle.mace.batch_size if self.config.oracle.mace else 32
        iterator = iter(structures)

        while True:
            chunk = list(islice(iterator, batch_size))
            if not chunk:
                break

            self._process_chunk(chunk)

            yield from chunk

    def _process_chunk(self, chunk: list[StructureMetadata]) -> None:
        to_compute: list[tuple[int, StructureMetadata, Atoms]] = []

        for i, s in enumerate(chunk):
            s.label_source = "mace"
            atoms = self._validate_and_extract_atoms(s)
            if atoms is not None:
                to_compute.append((i, s, atoms))

        if not to_compute:
            return

        if self.config.oracle.mock:
            self._mock_compute(to_compute)
        else:
            self._real_compute(to_compute)

    def _mock_compute(self, items: list[tuple[int, StructureMetadata, Atoms]]) -> None:
        for _, s, atoms in items:
            s.energy = -10.0
            s.forces = [[0.0, 0.0, 0.0] for _ in range(len(atoms))]
            s.status = StructureStatus.CALCULATED

    def _real_compute(self, items: list[tuple[int, StructureMetadata, Atoms]]) -> None:
        if self.mace_manager is None:
            msg = "MaceManager is None but mock is False"
            raise ConfigurationError(msg)

        try:
            atoms_list = [t[2] for t in items]
            results = self.mace_manager.compute_batch(atoms_list)

            for item, res_atoms in zip(items, results, strict=True):
                _, s, _ = item
                update_structure_metadata(s, res_atoms)
        except Exception:
            self.logger.exception("Batch computation failed")
            for _, s, _ in items:
                s.status = StructureStatus.FAILED

    def compute_uncertainty(
        self, structures: Iterable[StructureMetadata]
    ) -> Iterator[StructureMetadata]:
        """Compute uncertainty for a batch of structures."""
        self.logger.info("Computing uncertainty (MACE)")

        batch_size = 100
        iterator = iter(structures)

        while True:
            chunk = list(islice(iterator, batch_size))
            if not chunk:
                break

            self._process_uncertainty_chunk(chunk)

            yield from chunk

    def _process_uncertainty_chunk(self, chunk: list[StructureMetadata]) -> None:
        atoms_list = []
        valid_indices = []
        for i, s in enumerate(chunk):
            atoms = self._extract_atoms(s)
            if atoms:
                atoms_list.append(atoms)
                valid_indices.append(i)

        if not atoms_list:
            return

        uncertainties = self._get_uncertainties(atoms_list)

        for idx, unc in zip(valid_indices, uncertainties, strict=False):
            s = chunk[idx]
            if unc is not None:
                s.uncertainty_state = UncertaintyState(
                    gamma_mean=float(unc), gamma_max=float(unc)
                )

    def _get_uncertainties(self, atoms_list: list[Atoms]) -> list[float | None]:
        if self.config.oracle.mock:
            return [0.5] * len(atoms_list)

        if self.mace_manager:
            try:
                return [
                    float(x) for x in self.mace_manager.compute_uncertainty(atoms_list)
                ]
            except Exception:
                self.logger.exception("Failed to compute uncertainty batch")

        return [None] * len(atoms_list)
