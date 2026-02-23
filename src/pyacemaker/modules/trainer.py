"""Trainer (Pacemaker) module implementation."""

import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from uuid import UUID

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
        # 1. Prepare Dataset
        # Generator for valid structures - Streaming
        valid_structures = (s for s in dataset if s.energy is not None and s.forces is not None)

        # Create work directory
        work_dir = Path(tempfile.mkdtemp(prefix=CONSTANTS.TRAINER_TEMP_PREFIX_TRAIN))
        dataset_path = work_dir / CONSTANTS.default_training_file

        # Convert to Atoms and save (Streaming)
        # We use a mutable counter inside the generator context
        # This prevents loading anything into a list
        stats = {"count": 0}

        def counting_stream(structures: Iterable[StructureMetadata]) -> Iterator[Any]:
            for s in structures:
                stats["count"] += 1
                yield s

        # Use helper stream_metadata_to_atoms which uses metadata_to_atoms (now injects UUID)
        atoms_stream = stream_metadata_to_atoms(counting_stream(valid_structures))

        # save_iter consumes the generator completely
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

        # Remove internal config keys not used by pace_train directly
        # Delta learning is handled via baseline file, so remove it from params
        params.pop("delta_learning", None)
        # Parameters dict is internal/complex, not a CLI flag
        params.pop("parameters", None)
        # Mock flag is internal
        params.pop("mock", None)

        # If baseline file exists, pass it (assuming pace_train supports --baseline)
        if baseline_file:
            params["baseline"] = str(baseline_file)

        # Merge additional arguments (e.g. from delta learning workflow)
        params.update(kwargs)

        # 4. Train
        if self.trainer_config.mock:
            self.logger.info("Mock Mode: Skipping pace_train execution.")
            output_pot_path = work_dir / CONSTANTS.default_trainer_mock_potential_name
            output_pot_path.touch()
        else:
            initial_pot_path = initial_potential.path if initial_potential else None
            output_pot_path = self.wrapper.train(dataset_path, work_dir, params, initial_pot_path)

        # 5. Return Potential
        return Potential(
            path=output_pot_path,
            type=PotentialType.PACE,
            version=self.config.version,
            metrics={},
            parameters=self.trainer_config.model_dump(),
        )

    def select_active_set(
        self, candidates: Iterable[StructureMetadata], n_select: int
    ) -> ActiveSet:
        """Select active set."""
        work_dir = Path(tempfile.mkdtemp(prefix=CONSTANTS.TRAINER_TEMP_PREFIX_ACTIVE))
        candidates_path = work_dir / CONSTANTS.default_candidates_file

        # Save candidates (Streaming)
        # stream_metadata_to_atoms uses metadata_to_atoms which now injects UUID.
        self.dataset_manager.save_iter(
            stream_metadata_to_atoms(candidates), candidates_path
        )

        # Run selection
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
            # ActiveSetSelector.select now typically takes path and returns path.
            # Assuming it handles large files by passing path to CLI tool.
            selected_path = self.active_set_selector.select(candidates_path, n_select)

        # Process selected structures - Streaming Only
        # We only need IDs for the ActiveSet record if we are strict.
        # We process the result file lazily to extract IDs.
        selected_ids: list[UUID] = []

        # We must iterate to get IDs, but we discard objects immediately.
        for atoms in self.dataset_manager.load_iter(selected_path):
            uid_str = atoms.info.get("uuid")
            if uid_str:
                try:
                    selected_ids.append(UUID(uid_str))
                except ValueError:
                    self.logger.warning(f"Invalid UUID in selected structure: {uid_str}")

        return ActiveSet(
            structure_ids=selected_ids,
            structures=None,  # Enforce loading from path to prevent OOM
            dataset_path=selected_path,
            selection_criteria="max_vol",
        )

    # _metadata_to_atoms removed as stream_metadata_to_atoms/metadata_to_atoms is used

    def _generate_baseline(self, path: Path, type_: str) -> None:
        """Generate baseline potential file."""
        self.logger.info(f"Generating {type_} baseline potential at {path}")
        # Placeholder
        path.touch()
