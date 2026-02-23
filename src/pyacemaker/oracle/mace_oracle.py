"""MACE Surrogate Oracle Module."""

from collections.abc import Iterable, Iterator

from ase import Atoms
from loguru import logger

from pyacemaker.core.base import Metrics, ModuleResult
from pyacemaker.core.config import MaceConfig, PYACEMAKERConfig
from pyacemaker.core.interfaces import Oracle, UncertaintyModel
from pyacemaker.core.utils import update_structure_metadata
from pyacemaker.domain_models.models import (
    StructureMetadata,
    StructureStatus,
    UncertaintyState,
)
from pyacemaker.oracle.mace_manager import MaceManager


class MaceSurrogateOracle(Oracle, UncertaintyModel):
    """Oracle wrapping MACE potential for prediction and uncertainty."""

    def __init__(self, config: PYACEMAKERConfig | MaceConfig) -> None:
        """Initialize the MACE Oracle."""
        # BaseModule expects PYACEMAKERConfig
        self.mace_config: MaceConfig

        if isinstance(config, PYACEMAKERConfig):
            super().__init__(config)
            if config.oracle.mace:
                self.mace_config = config.oracle.mace
            else:
                self.logger.warning("MACE config missing, using defaults for Oracle.")
                self.mace_config = MaceConfig(model_path="medium")
        else:
            self.config = config  # type: ignore[assignment]
            self.mace_config = config
            self.logger = logger.bind(name="MaceSurrogateOracle")

        self.manager = MaceManager(self.mace_config)

        # Load model immediately? Or lazy?
        # Manager loads lazily.

    def run(self) -> ModuleResult:
        """Execute default oracle task."""
        return ModuleResult(
            status="success",
            metrics=Metrics(message="MaceSurrogateOracle ready")  # type: ignore[call-arg]
        )

    def compute_batch(self, structures: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
        """Compute energy/forces for a batch of structures."""
        # MACE supports batching natively if we pass list of atoms.
        # But here we stream.
        # We can buffer into batches and process.

        batch_size = self.mace_config.batch_size
        buffer: list[StructureMetadata] = []

        for s in structures:
            buffer.append(s)
            if len(buffer) >= batch_size:
                yield from self._process_batch(buffer)
                buffer.clear()

        if buffer:
            yield from self._process_batch(buffer)

    def compute_uncertainty(
        self, structures: Iterable[StructureMetadata]
    ) -> Iterator[StructureMetadata]:
        """Compute uncertainty for a batch of structures."""
        # Similar batching logic
        batch_size = self.mace_config.batch_size
        buffer: list[StructureMetadata] = []

        for s in structures:
            buffer.append(s)
            if len(buffer) >= batch_size:
                yield from self._process_uncertainty_batch(buffer)
                buffer.clear()

        if buffer:
            yield from self._process_uncertainty_batch(buffer)

    def _process_batch(self, batch: list[StructureMetadata]) -> Iterator[StructureMetadata]:
        """Process a batch for prediction."""
        # For simple MACE usage, we might loop if batching not exposed in manager.
        # MaceManager.compute takes single structure.
        # Ideally we update MaceManager to support batch.
        # But for now, loop.

        for s in batch:
            if "atoms" not in s.features:
                continue

            try:
                atoms = s.features["atoms"]
                result_atoms = self.manager.compute(atoms)
                update_structure_metadata(s, result_atoms)
                s.label_source = "mace"
            except Exception:
                self.logger.exception(f"MACE prediction failed for {s.id}")
                s.status = StructureStatus.FAILED

            yield s

    def _process_uncertainty_batch(self, batch: list[StructureMetadata]) -> Iterator[StructureMetadata]:
        """Process a batch for uncertainty."""
        atoms_list: list[Atoms] = []
        valid_indices = []

        for i, s in enumerate(batch):
            if "atoms" in s.features:
                atoms_list.append(s.features["atoms"])
                valid_indices.append(i)

        if not atoms_list:
            # Yield originals unmodified if no valid atoms
            yield from batch
            return

        try:
            uncertainties = self.manager.compute_uncertainty(atoms_list)
        except Exception:
            self.logger.exception("MACE uncertainty computation failed")
            yield from batch
            return

        # Assign back
        for idx, unc in zip(valid_indices, uncertainties, strict=True):
            s = batch[idx]
            # Update UncertaintyState
            if s.uncertainty_state is None:
                s.uncertainty_state = UncertaintyState()

            s.uncertainty_state.gamma_max = unc
            # We assume gamma_max maps to the uncertainty value returned

        yield from batch
