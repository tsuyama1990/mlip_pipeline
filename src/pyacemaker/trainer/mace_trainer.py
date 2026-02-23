"""MACE Trainer implementation."""

import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.utils import stream_metadata_to_atoms
from pyacemaker.domain_models.models import (
    ActiveSet,
    Potential,
    PotentialType,
    StructureMetadata,
)
from pyacemaker.oracle.dataset import DatasetManager
from pyacemaker.oracle.mace_manager import MaceManager
from pyacemaker.trainer.base import BaseTrainer


class MaceTrainer(BaseTrainer):
    """MACE trainer implementation."""

    def __init__(self, config: PYACEMAKERConfig) -> None:
        """Initialize MaceTrainer."""
        super().__init__(config)
        # Assuming MACE config is in oracle.mace for now
        if config.oracle.mace:
            self.mace_manager: MaceManager | None = MaceManager(config.oracle.mace)
        else:
            self.mace_manager = None
        self.dataset_manager = DatasetManager()

    def run(self) -> Any:
        """Run the trainer."""
        self.logger.info("Running MaceTrainer")
        return {"status": "success"}

    def _validate_train_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Validate and sanitize training arguments."""
        # Allowlist of parameters we expect to pass to MaceManager
        # MaceManager has its own strict validation against _ALLOWED_TRAIN_PARAMS
        # Here we just ensure we don't pass anything completely unexpected if needed
        # For now, we rely on MaceManager's validation but could add Pydantic model here.
        return kwargs

    def train(
        self,
        dataset: Iterable[StructureMetadata],
        initial_potential: Potential | None = None,
        **kwargs: Any,
    ) -> Potential:
        """Train or Fine-tune MACE model.

        Args:
            dataset: Streaming iterator of structures.
            initial_potential: Optional base potential for fine-tuning.
            **kwargs: Training parameters (e.g. epochs, batch_size).

        Returns:
            Potential: The trained MACE potential.
        """
        if not self.mace_manager:
            msg = "MACE Manager not initialized. Check config."
            raise ValueError(msg)

        # 1. Prepare Dataset
        # Use a safe temporary directory
        work_dir = Path(tempfile.mkdtemp(prefix="mace_train_"))
        dataset_path = work_dir / "training_data.xyz"

        # Filter valid structures
        valid_dataset = (
            s for s in dataset
            if s.energy is not None and s.forces is not None
        )

        # Save to file using streaming to prevent OOM
        self.dataset_manager.save_iter(stream_metadata_to_atoms(valid_dataset), dataset_path)

        # 2. Train
        # Params from config or specific distillation params
        params: dict[str, Any] = {"max_num_epochs": 50}  # Default
        if self.config.distillation and self.config.distillation.step3_mace_finetune:
            mace_conf = self.config.distillation.step3_mace_finetune
            params["max_num_epochs"] = mace_conf.epochs

        if initial_potential:
            params["foundation_model"] = str(initial_potential.path)

        # Merge extra params (optional)
        safe_kwargs = self._validate_train_kwargs(kwargs)
        params.update(safe_kwargs)

        # Map common aliases
        if "epochs" in params:
            params["max_num_epochs"] = params.pop("epochs")

        output_path = self.mace_manager.train(dataset_path, work_dir, params)

        return Potential(
            path=output_path,
            type=PotentialType.MACE,
            version=self.config.version,
            metrics={},
            parameters=params,
        )

    def select_active_set(
        self, candidates: Iterable[StructureMetadata], n_select: int
    ) -> ActiveSet:
        """Select active set.

        Currently not implemented for MACE as the active learning logic resides in the Orchestrator/ActiveLearner.
        """
        # Return an empty ActiveSet or raise error.
        # The Orchestrator handles MACE AL via ActiveLearner module directly in Cycle 02 logic.
        # So this might technically be unreachable in current workflow,
        # but for interface compliance we raise or return empty.
        msg = "MaceTrainer.select_active_set is not implemented."
        raise NotImplementedError(msg)
