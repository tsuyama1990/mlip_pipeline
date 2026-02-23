"""MACE Oracle implementation."""

from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING

from ase import Atoms

from pyacemaker.core.base import ModuleResult
from pyacemaker.core.config import CONSTANTS, PYACEMAKERConfig
from pyacemaker.core.exceptions import ConfigurationError
from pyacemaker.core.interfaces import UncertaintyModel
from pyacemaker.core.utils import update_structure_metadata, validate_structure_integrity
from pyacemaker.domain_models.models import (
    StructureMetadata,
    StructureStatus,
    UncertaintyState,
)
from pyacemaker.oracle.base_oracle import BaseOracle
from pyacemaker.oracle.mace_manager import MaceManager

if TYPE_CHECKING:
    pass


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
            self.mace_manager: MaceManager | None = None
        else:
            self.mace_manager = MaceManager(config.oracle.mace)

    def run(self) -> ModuleResult:
        """Run the oracle (batch processing)."""
        self.logger.info("Running MaceSurrogateOracle")
        return ModuleResult(status="success")

    def update_model(self, path: Path) -> None:
        """Update the MACE model path."""
        self.logger.info(f"Updating MACE model to {path}")
        if self.config.oracle.mock:
            return

        if self.mace_manager:
            self.mace_manager.update_model_path(path)

    def predict_batch(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        """Efficiently process a list of structures (Batch Prediction).

        This method updates the energy, forces, and stress fields of the input structures in-place.
        Returns the updated list.
        """
        self.logger.info(f"Predicting batch of {len(structures)} structures (MACE)")

        # In Mock mode, we just fill dummy data
        if self.config.oracle.mock:
            for s in structures:
                s.energy = -10.0
                # Use number of atoms if available, else generic
                atoms = self._extract_atoms(s)
                n_atoms = len(atoms) if atoms else 1
                s.forces = [[0.0, 0.0, 0.0] for _ in range(n_atoms)]
                s.stress = [0.0] * 6
                s.status = StructureStatus.CALCULATED
                s.label_source = "mace"
            return structures

        if not self.mace_manager:
            # Should not happen if not mock
            msg = "MaceManager is not initialized."
            raise ConfigurationError(msg)

        # Batch processing loop
        # Although MaceManager.compute handles one at a time, we wrap it here.
        # Future optimization: Implement true batching in MaceManager and call it here.
        for s in structures:
            try:
                # Basic validation
                self.validate_structure(s)
                atoms = self._extract_atoms(s)
                if not atoms:
                    continue

                # Compute
                result_atoms = self.mace_manager.compute(atoms)

                # Update metadata
                update_structure_metadata(s, result_atoms)
                s.label_source = "mace"
                s.status = StructureStatus.CALCULATED

            except Exception:
                self.logger.exception(f"Failed to predict structure {s.id}")
                s.status = StructureStatus.FAILED

        return structures

    def compute_batch(self, structures: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
        """Compute energy/forces for a batch of structures (Streaming Interface)."""
        self.logger.info("Computing batch of structures (MACE Streaming)")

        chunk_size = CONSTANTS.oracle_chunk_size
        iterator = iter(structures)

        while True:
            chunk = list(islice(iterator, chunk_size))
            if not chunk:
                break

            # Process chunk using predict_batch
            yield from self.predict_batch(chunk)

    def _collect_valid_batch(
        self, chunk: list[StructureMetadata]
    ) -> tuple[list[Atoms], list[int]]:
        """Collect valid atoms from a chunk of structures."""
        atoms_list: list[Atoms] = []
        valid_indices: list[int] = []

        for i, s in enumerate(chunk):
            try:
                validate_structure_integrity(s)
                atoms = self._extract_atoms(s)
                if atoms:
                    atoms_list.append(atoms)
                    valid_indices.append(i)
            except (ValueError, TypeError):
                self.logger.warning(f"Skipping invalid structure {s.id} in uncertainty computation")

        return atoms_list, valid_indices

    def _process_uncertainty_chunk(self, chunk: list[StructureMetadata]) -> None:
        """Process a single chunk for uncertainty computation."""
        atoms_list, valid_indices = self._collect_valid_batch(chunk)

        if not atoms_list:
            return

        uncertainties: list[float] = []
        if self.config.oracle.mock:
            uncertainties = [0.5] * len(atoms_list)  # Mock uncertainty
        elif self.mace_manager:
            try:
                uncertainties = self.mace_manager.compute_uncertainty(atoms_list)
            except Exception:
                self.logger.exception("Failed to compute uncertainty batch")
                return

        # Assign back if we have results
        if len(uncertainties) == len(valid_indices):
            for idx, unc in zip(valid_indices, uncertainties, strict=False):
                s = chunk[idx]
                s.uncertainty_state = UncertaintyState(gamma_mean=unc, gamma_max=unc)

    def compute_uncertainty(
        self, structures: Iterable[StructureMetadata]
    ) -> Iterator[StructureMetadata]:
        """Compute uncertainty for a batch of structures."""
        self.logger.info("Computing uncertainty (MACE)")

        chunk_size = CONSTANTS.oracle_chunk_size
        iterator = iter(structures)

        while True:
            chunk = list(islice(iterator, chunk_size))
            if not chunk:
                break

            self._process_uncertainty_chunk(chunk)

            yield from chunk
