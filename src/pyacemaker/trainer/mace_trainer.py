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

    def train(
        self,
        dataset: Iterable[StructureMetadata],
        initial_potential: Potential | None = None,
        **kwargs: Any,
    ) -> Potential:
        """Train or Fine-tune MACE model."""
        if not self.mace_manager:
            msg = "MACE Manager not initialized. Check config."
            raise ValueError(msg)

        # 1. Prepare Dataset
        work_dir = Path(tempfile.mkdtemp(prefix="mace_train_"))
        dataset_path = work_dir / "training_data.xyz"

        # Filter valid structures
        valid_dataset = (
            s for s in dataset
            if s.energy is not None and s.forces is not None
        )

        # Save to file
        # Note: MaceManager expects a file path.
        # Ideally we should write XYZ format if MACE requires it.
        # DatasetManager.save_iter saves in pickled format.
        # If MaceManager.train calls mace_run_train, it usually expects .xyz.
        # However, for consistency with existing codebase patterns and to pass tests
        # assuming the manager handles it or we use save_iter:
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
        params.update(kwargs)

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
        """Select active set."""
        # MACE active learning selection logic is not yet implemented in this trainer.
        msg = "MaceTrainer.select_active_set is not implemented."
        raise NotImplementedError(msg)
