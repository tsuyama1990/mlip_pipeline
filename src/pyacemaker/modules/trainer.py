"""Trainer (Pacemaker) module implementation."""

import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pyacemaker.core.config import CONSTANTS, PYACEMAKERConfig
from pyacemaker.core.interfaces import Trainer
from pyacemaker.core.utils import validate_structure_integrity
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
        self,
        dataset: Iterable[StructureMetadata],
        initial_potential: Potential | None = None,
    ) -> Potential:
        """Train a potential (Streaming)."""
        # 1. Prepare Dataset
        # Generator for valid structures
        valid_structures = (
            s for s in dataset if s.energy is not None and s.forces is not None
        )

        # Create work directory
        work_dir = Path(tempfile.mkdtemp(prefix=CONSTANTS.TRAINER_TEMP_PREFIX_TRAIN))
        dataset_path = work_dir / "training_set.pckl.gzip"

        # Convert to Atoms and save (Streaming)
        # We need to count to ensure we have data, but save_iter consumes.
        # We can wrap the generator to count.
        count = 0

        def counting_wrapper(gen: Iterable[StructureMetadata]) -> Iterator[Any]:
            nonlocal count
            for s in gen:
                count += 1
                yield self._metadata_to_atoms(s)

        self.dataset_manager.save_iter(counting_wrapper(valid_structures), dataset_path)

        if count == 0:
            msg = "No valid structures with energy and forces found for training."
            raise ValueError(msg)

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
        # Parameters dict is internal/complex, not a CLI flag
        params.pop("parameters", None)
        # Mock flag is internal
        params.pop("mock", None)

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

    def select_active_set(
        self, candidates: Iterable[StructureMetadata], n_select: int
    ) -> ActiveSet:
        """Select active set."""
        work_dir = Path(tempfile.mkdtemp(prefix=CONSTANTS.TRAINER_TEMP_PREFIX_ACTIVE))
        candidates_path = work_dir / "candidates.pckl.gzip"

        # Save candidates (Streaming)
        atoms_gen = (self._metadata_to_atoms(s) for s in candidates)
        self.dataset_manager.save_iter(atoms_gen, candidates_path)

        selected_structures_list: list[StructureMetadata] = []

        # Run selection
        if self.trainer_config.mock:
            self.logger.info("Mock Mode: Skipping pace_activeset execution.")
            # We can't easily slice a generator without consuming it or caching.
            # But in mock mode, we usually just want to test flow.
            # We'll reload the saved candidates and take first n_select
            selected_path = work_dir / "selected.pckl.gzip"
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

        # Load selected structures from file to reconstruct metadata
        for atoms in self.dataset_manager.load_iter(selected_path):
            uid_str = atoms.info.get("uuid")
            uid = UUID(uid_str) if uid_str else uuid4()

            # Reconstruct minimal metadata
            meta = StructureMetadata(
                id=uid,
                features={"atoms": atoms},
                energy=atoms.info.get("energy"),
                # Forces/Stress reconstruction if available
            )
            if "forces" in atoms.arrays:
                meta.forces = atoms.arrays["forces"].tolist()
            if "stress" in atoms.info:
                stress_val = atoms.info["stress"]
                meta.stress = (
                    stress_val.tolist()
                    if hasattr(stress_val, "tolist")
                    else stress_val
                )

            selected_structures_list.append(meta)

        selected_ids = [s.id for s in selected_structures_list]

        return ActiveSet(
            structure_ids=selected_ids,
            structures=selected_structures_list,
            selection_criteria="max_vol",
        )

    def _metadata_to_atoms(self, metadata: StructureMetadata) -> Any:
        """Convert StructureMetadata to ASE Atoms."""
        validate_structure_integrity(metadata)
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
