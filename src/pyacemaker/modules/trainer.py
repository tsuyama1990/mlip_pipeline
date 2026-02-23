"""Trainer (Pacemaker) module implementation."""

import shutil
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pyacemaker.core.config import CONSTANTS, PYACEMAKERConfig
from pyacemaker.core.interfaces import Trainer
from pyacemaker.core.utils import (
    stream_metadata_to_atoms,
)
from pyacemaker.domain_models.models import (
    ActiveSet,
    Potential,
    PotentialType,
    StructureMetadata,
)
from pyacemaker.oracle.dataset import DatasetManager
from pyacemaker.trainer.active_set import ActiveSetSelector
from pyacemaker.trainer.mace_trainer import MaceTrainer
from pyacemaker.trainer.wrapper import PacemakerWrapper

try:
    from ase import Atoms
except ImportError:
    Atoms = Any  # type: ignore[assignment,misc]

__all__ = ["MaceTrainer", "PacemakerTrainer"]


class PacemakerTrainer(Trainer):
    """Pacemaker trainer implementation."""

    def __init__(self, config: PYACEMAKERConfig) -> None:
        """Initialize PacemakerTrainer."""
        super().__init__(config)
        self.trainer_config = config.trainer
        self.wrapper = PacemakerWrapper()
        self.active_set_selector = ActiveSetSelector(wrapper=self.wrapper)
        # Expose dataset_manager as public attribute for tests
        self.dataset_manager = DatasetManager()

    def run(self) -> Any:
        """Run the trainer (Placeholder for interface compliance)."""
        self.logger.info("Running PacemakerTrainer")
        return {"status": "success"}

    def train(
        self,
        dataset: Iterable[StructureMetadata],
        initial_potential: Potential | None = None,
        **kwargs: Any,
    ) -> Potential:
        """Train a potential (Streaming)."""
        # Ensure persistent models directory exists
        models_dir = self.config.project.root_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix=CONSTANTS.TRAINER_TEMP_PREFIX_TRAIN) as temp_dir_str:
            work_dir = Path(temp_dir_str)
            dataset_path = work_dir / CONSTANTS.default_training_file

            # Prepare streaming generator with validation and counting
            stats = {"count": 0}

            def valid_counting_stream(structures: Iterable[StructureMetadata]) -> Iterator[Any]:
                for s in structures:
                    if s.energy is not None and s.forces is not None:
                        stats["count"] += 1
                        yield s

            # Use helper stream_metadata_to_atoms
            atoms_stream = stream_metadata_to_atoms(valid_counting_stream(dataset))
            self.dataset_manager.save_iter(atoms_stream, dataset_path)

            if stats["count"] == 0:
                msg = "No valid structures with energy and forces found for training."
                raise ValueError(msg)

            # 2. Configure Delta Learning (Baseline)
            baseline_file = None
            if self.trainer_config.delta_learning in ("zbl", "lj"):
                baseline_file = (
                    work_dir
                    / f"{self.trainer_config.delta_learning}{CONSTANTS.default_trainer_baseline_suffix}"
                )
                self._generate_baseline(baseline_file, self.trainer_config.delta_learning)

            # 3. Prepare Params
            params = self.trainer_config.model_dump(exclude={"potential_type"})
            params.pop("delta_learning", None)
            params.pop("parameters", None)
            params.pop("mock", None)

            if baseline_file:
                params["baseline"] = str(baseline_file)

            params.update(kwargs)

            # 4. Train
            if self.trainer_config.mock:
                self.logger.info("Mock Mode: Skipping pace_train execution.")
                output_pot_path = work_dir / CONSTANTS.default_trainer_mock_potential_name
                output_pot_path.touch()
            else:
                initial_pot_path = initial_potential.path if initial_potential else None
                output_pot_path = self.wrapper.train(dataset_path, work_dir, params, initial_pot_path)

            # Persist the model
            unique_name = f"pace_model_{uuid4().hex[:8]}.yace"
            final_path = models_dir / unique_name

            if output_pot_path.exists():
                shutil.copy2(output_pot_path, final_path)
            elif self.trainer_config.mock:
                final_path.touch()
            else:
                msg = f"Model not found at {output_pot_path}"
                raise FileNotFoundError(msg)

        # 5. Return Potential
        return Potential(
            path=final_path,
            type=PotentialType.PACE,
            version=self.config.version,
            metrics={},
            parameters=self.trainer_config.model_dump(),
        )

    def select_active_set(
        self, candidates: Iterable[StructureMetadata], n_select: int
    ) -> ActiveSet:
        """Select active set."""
        with tempfile.TemporaryDirectory(prefix=CONSTANTS.TRAINER_TEMP_PREFIX_ACTIVE) as temp_dir_str:
            work_dir = Path(temp_dir_str)
            candidates_path = work_dir / CONSTANTS.default_candidates_file

            self.dataset_manager.save_iter(
                stream_metadata_to_atoms(candidates), candidates_path
            )

            if self.trainer_config.mock:
                self.logger.info("Mock Mode: Skipping pace_activeset execution.")
                selected_path = work_dir / CONSTANTS.default_selected_file
                # Limit generator
                reloaded_gen = self.dataset_manager.load_iter(candidates_path)

                def limited_gen() -> Iterator[Any]:
                    for i, atoms in enumerate(reloaded_gen):
                        if i >= n_select:
                            break
                        yield atoms

                self.dataset_manager.save_iter(limited_gen(), selected_path)
            else:
                selected_path = self.active_set_selector.select(candidates_path, n_select)

            # Persist active set
            data_dir = self.config.project.root_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            final_set_path = data_dir / f"active_set_{uuid4().hex[:8]}.xyz"

            if selected_path.exists():
                shutil.copy2(selected_path, final_set_path)
            else:
                msg = f"Selected set not found at {selected_path}"
                raise FileNotFoundError(msg)

            # Process selected structures - Streaming Only
            selected_ids: list[UUID] = []

            for atoms in self.dataset_manager.load_iter(final_set_path):
                uid_str = atoms.info.get("uuid")
                if uid_str:
                    try:
                        selected_ids.append(UUID(uid_str))
                    except ValueError:
                        self.logger.warning(f"Invalid UUID in selected structure: {uid_str}")

        return ActiveSet(
            structure_ids=selected_ids,
            structures=None,
            dataset_path=final_set_path,
            selection_criteria="max_vol",
        )

    # _metadata_to_atoms removed as stream_metadata_to_atoms/metadata_to_atoms is used

    def _generate_baseline(self, path: Path, type_: str) -> None:
        """Generate baseline potential file."""
        self.logger.info(f"Generating {type_} baseline potential at {path}")
        # Placeholder
        path.touch()
