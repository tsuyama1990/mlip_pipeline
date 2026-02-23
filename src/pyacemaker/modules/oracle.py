"""Oracle (DFT) module implementation."""

import concurrent.futures
import random
import secrets
from collections.abc import Iterable, Iterator
from typing import Any

from ase import Atoms

from pyacemaker.core.base import ModuleResult
from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.exceptions import PYACEMAKERError
from pyacemaker.core.utils import update_structure_metadata
from pyacemaker.domain_models.models import (
    StructureMetadata,
    StructureStatus,
)
from pyacemaker.oracle.base_oracle import BaseOracle
from pyacemaker.oracle.mace_oracle import MaceSurrogateOracle
from pyacemaker.oracle.manager import DFTManager

__all__ = ["DFTOracle", "MaceSurrogateOracle", "MockOracle"]


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

    def _compute_lj(self, atoms: Atoms) -> tuple[float, list[list[float]]]:
        """Compute simplified Lennard-Jones potential."""
        # Simple LJ parameters (mocking Fe)
        # epsilon = 1.0 eV, sigma = 2.0 A
        epsilon = 1.0
        sigma = 2.0

        positions = atoms.get_positions()
        n_atoms = len(atoms)

        energy = 0.0
        forces = [[0.0, 0.0, 0.0] for _ in range(n_atoms)]

        if n_atoms < 50:
            import numpy as np
            from scipy.spatial.distance import pdist

            # Pairwise distances
            dists = pdist(positions)
            # Avoid singularity
            dists[dists < 0.1] = 0.1

            # Lennard-Jones: 4*eps * ((sigma/r)^12 - (sigma/r)^6)
            sr6 = (sigma / dists) ** 6
            sr12 = sr6 ** 2
            pot = 4 * epsilon * (sr12 - sr6)
            energy = np.sum(pot)
        else:
            energy = -100.0 # Baseline

        # Add deterministic noise based on positions
        pos_hash = hash(positions.tobytes())
        rng = random.Random(pos_hash)  # noqa: S311

        noise_e = rng.uniform(-1.0, 1.0)
        energy += noise_e

        # Forces are just noise
        forces = [[rng.uniform(-0.1, 0.1) for _ in range(3)] for _ in range(n_atoms)]

        return energy, forces

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

            # Compute LJ + Noise
            energy, forces = self._compute_lj(atoms)

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

    def _submit_structure(
        self,
        s: StructureMetadata,
        executor: concurrent.futures.ThreadPoolExecutor,
        futures: dict[concurrent.futures.Future[Any], StructureMetadata],
    ) -> bool:
        """Submit a single structure to the executor."""
        atoms = self._validate_and_extract_atoms(s)
        if atoms:
            try:
                future = executor.submit(self.dft_manager.compute, atoms)
                futures[future] = s
                return True
            except Exception:
                self.logger.exception(f"Failed to submit task for {s.id}")
                s.status = StructureStatus.FAILED
                return False
        return False

    def compute_batch(self, structures: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
        """Compute energy/forces for a batch of structures."""
        self.logger.info("Computing batch of structures (DFT Parallel Streaming)")

        max_workers = self.config.oracle.dft.max_workers
        buffer_size = max_workers * 2
        iterator = iter(structures)
        futures: dict[concurrent.futures.Future[Any], StructureMetadata] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Initial fill
            for _ in range(buffer_size):
                try:
                    s = next(iterator)
                except StopIteration:
                    break

                if not self._submit_structure(s, executor, futures):
                    yield s

            # Process loop
            while futures:
                done, _ = concurrent.futures.wait(
                    futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                )

                for future in done:
                    s = futures.pop(future)
                    try:
                        result_atoms = future.result()
                        update_structure_metadata(s, result_atoms)
                    except Exception:
                        self.logger.exception(f"Error computing structure {s.id}")
                        s.status = StructureStatus.FAILED
                    yield s

                # Refill
                while len(futures) < buffer_size:
                    try:
                        s = next(iterator)
                    except StopIteration:
                        break

                    if not self._submit_structure(s, executor, futures):
                        yield s
