"""MACE Trainer implementation."""

import shutil
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

from pyacemaker.core.config import CONSTANTS, PYACEMAKERConfig
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

        # Ensure persistent models directory exists
        models_dir = self.config.project.root_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Use temporary directory context for cleanup
        with tempfile.TemporaryDirectory(prefix="mace_train_") as temp_dir_str:
            work_dir = Path(temp_dir_str)
            dataset_path = work_dir / "training_data.xyz"

            # Filter valid structures using generator function
            def valid_stream(data: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
                for s in data:
                    if s.energy is not None and s.forces is not None:
                        yield s

            valid_dataset_iter = valid_stream(dataset)

            # Save to file using streaming to prevent OOM
            # stream_metadata_to_atoms returns a generator, save_iter consumes it lazily.
            self.dataset_manager.save_iter(
                stream_metadata_to_atoms(valid_dataset_iter), dataset_path
            )

            # 2. Train
            # Params from config or specific distillation params
            params: dict[str, Any] = {
                "max_num_epochs": CONSTANTS.mace_default_max_epochs
            }
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

            # Persist the model
            unique_name = f"mace_model_{uuid4().hex[:8]}.model"
            final_path = models_dir / unique_name

            if output_path.exists():
                shutil.copy2(output_path, final_path)
            else:
                # Should have been handled by mace_manager but double check
                msg = f"Model not found at {output_path}"
                raise FileNotFoundError(msg)

        return Potential(
            path=final_path,
            type=PotentialType.MACE,
            version=self.config.version,
            metrics={},
            parameters=params,
        )

    def select_active_set(
        self, candidates: Iterable[StructureMetadata], n_select: int
    ) -> ActiveSet:
        """Select active set stub.

        MACE active learning logic is handled by the Orchestrator/ActiveLearner module.
        This method exists for interface compliance.
        """
        self.logger.warning("MaceTrainer.select_active_set called but MACE AL uses external logic.")
        return ActiveSet(
            structure_ids=[],
            structures=None,
            dataset_path=None,
            selection_criteria="external_mace_al"
        )
