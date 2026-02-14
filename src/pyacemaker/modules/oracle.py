"""Oracle (DFT) module implementation."""

import random

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
            # Mock results with slight randomness
            energy = -100.0 + random.uniform(-1.0, 1.0)  # noqa: S311
            forces = [[random.uniform(-0.1, 0.1) for _ in range(3)]]  # noqa: S311

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

    def compute_batch(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        """Compute energy/forces for a batch of structures."""
        self.logger.info(f"Computing batch of {len(structures)} structures (DFT)")

        # Extract atoms from metadata
        atoms_list: list[Atoms] = []
        mapping: list[int] = []

        for i, s in enumerate(structures):
            # Check if atoms object is attached to features
            # In a real pipeline, we might need to load from file path
            # But for now we assume features["atoms"] holds the object
            atoms = s.features.get("atoms")
            if isinstance(atoms, Atoms):
                atoms_list.append(atoms)
                mapping.append(i)
            else:
                self.logger.warning(f"Structure {s.id} has no valid 'atoms' feature. Skipping.")

        if not atoms_list:
            self.logger.warning("No valid atoms found in batch.")
            return structures

        # Run DFT via manager
        results = self.dft_manager.compute_batch(atoms_list)

        # Update metadata with results
        for idx, result_atoms in zip(mapping, results, strict=True):
            s = structures[idx]
            if result_atoms is not None:
                s.status = StructureStatus.CALCULATED
                # Extract properties safely
                try:
                    # We use type: ignore because ASE types are often missing
                    s.features["energy"] = result_atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                    s.features["forces"] = result_atoms.get_forces().tolist()  # type: ignore[no-untyped-call]
                    s.features["stress"] = result_atoms.get_stress().tolist()  # type: ignore[no-untyped-call]
                    # Update atoms object in features (e.g. relaxed structure)
                    s.features["atoms"] = result_atoms
                except Exception:
                    # Catch all exceptions during extraction (e.g., property missing)
                    self.logger.exception(f"Failed to extract properties for {s.id}")
                    s.status = StructureStatus.FAILED
            else:
                s.status = StructureStatus.FAILED

        return structures
