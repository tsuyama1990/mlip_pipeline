"""Trainer (Pacemaker) module implementation."""

import tempfile
from pathlib import Path
from typing import Any

from pyacemaker.core.config import CONSTANTS, PYACEMAKERConfig
from pyacemaker.core.interfaces import Trainer
from pyacemaker.domain_models.models import (
    ActiveSet,
    Potential,
    PotentialType,
    StructureMetadata,
)
from pyacemaker.oracle.dataset import DatasetManager
from pyacemaker.trainer.active_set import ActiveSetSelector
from pyacemaker.trainer.wrapper import PacemakerWrapper


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
        self, dataset: list[StructureMetadata], initial_potential: Potential | None = None
    ) -> Potential:
        """Train a potential."""
        # 1. Prepare Dataset
        valid_structures = [s for s in dataset if s.energy is not None and s.forces is not None]
        if not valid_structures:
            msg = "No valid structures with energy and forces found for training."
            raise ValueError(msg)

        # Create work directory
        work_dir = Path(tempfile.mkdtemp(prefix=CONSTANTS.TRAINER_TEMP_PREFIX_TRAIN))
        dataset_path = work_dir / "training_set.pckl.gzip"

        # Convert to Atoms and save
        atoms_list = [self._metadata_to_atoms(s) for s in valid_structures]
        self.dataset_manager.save_iter(atoms_list, dataset_path)

        # 2. Configure Delta Learning (Baseline)
        baseline_file = None
        if self.trainer_config.delta_learning in ("zbl", "lj"):
            baseline_file = work_dir / f"{self.trainer_config.delta_learning}_baseline.yace"
            self._generate_baseline(baseline_file, self.trainer_config.delta_learning)

        # 3. Prepare Params
        params = self.trainer_config.model_dump(exclude={"potential_type"})

        # Remove internal config keys not used by pace_train directly
        # Delta learning is handled via baseline file, so remove it from params
        params.pop("delta_learning", None)

        # If baseline file exists, pass it (assuming pace_train supports --baseline)
        if baseline_file:
            params["baseline"] = str(baseline_file)

        # 4. Train
        if self.trainer_config.mock:
            self.logger.info("Mock Mode: Skipping pace_train execution.")
            output_pot_path = work_dir / "mock_potential.yace"
            output_pot_path.touch()
        else:
            initial_pot_path = initial_potential.path if initial_potential else None
            output_pot_path = self.wrapper.train(dataset_path, work_dir, params, initial_pot_path)

        # 5. Return Potential
        return Potential(
            path=output_pot_path,
            type=PotentialType.PACE,
            version="1.0",  # TODO: Implement proper versioning
            metrics={},  # TODO: Parse metrics from logs
            parameters=self.trainer_config.model_dump(),
        )

    def select_active_set(self, candidates: list[StructureMetadata], n_select: int) -> ActiveSet:
        """Select active set."""
        work_dir = Path(tempfile.mkdtemp(prefix=CONSTANTS.TRAINER_TEMP_PREFIX_ACTIVE))
        candidates_path = work_dir / "candidates.pckl.gzip"

        # Save candidates
        atoms_list = [self._metadata_to_atoms(s) for s in candidates]
        self.dataset_manager.save_iter(atoms_list, candidates_path)

        # Run selection
        if self.trainer_config.mock:
            self.logger.info("Mock Mode: Skipping pace_activeset execution.")
            # Just pick first n_select as dummy selection
            selected_candidates = candidates[:n_select]
            selected_path = work_dir / "selected.pckl.gzip"
            atoms_list_selected = [self._metadata_to_atoms(s) for s in selected_candidates]
            self.dataset_manager.save_iter(atoms_list_selected, selected_path)
        else:
            selected_path = self.active_set_selector.select(candidates_path, n_select)

        # Load selected to get IDs
        selected_ids = []
        for atoms in self.dataset_manager.load_iter(selected_path):
            if "uuid" in atoms.info:
                from uuid import UUID

                selected_ids.append(UUID(atoms.info["uuid"]))
            else:
                self.logger.warning("Selected structure missing UUID in info")

        return ActiveSet(
            structure_ids=selected_ids,
            selection_criteria="max_vol",
        )

    def _metadata_to_atoms(self, metadata: StructureMetadata) -> Any:
        """Convert StructureMetadata to ASE Atoms."""
        atoms = metadata.features.get("atoms")
        if atoms is None:
            msg = f"Structure {metadata.id} does not contain 'atoms' feature."
            raise ValueError(msg)

        # Create a copy to avoid modifying original
        atoms = atoms.copy()

        # Inject UUID
        atoms.info["uuid"] = str(metadata.id)

        # Inject Energy/Forces/Stress if available (overwrite calc results)
        if metadata.energy is not None:
            atoms.info["energy"] = metadata.energy
        if metadata.forces is not None:
            # arrays expects numpy array or list
            atoms.arrays["forces"] = metadata.forces
        if metadata.stress is not None:
            atoms.info["stress"] = metadata.stress

        return atoms

    def _generate_baseline(self, path: Path, type_: str) -> None:
        """Generate baseline potential file."""
        self.logger.info(f"Generating {type_} baseline potential at {path}")
        # Placeholder
        path.touch()
