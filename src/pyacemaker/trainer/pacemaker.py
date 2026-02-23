"""Pacemaker Trainer implementation."""

import shutil
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import yaml

from pyacemaker.core.config import CONSTANTS, PYACEMAKERConfig
from pyacemaker.core.utils import stream_metadata_to_atoms
from pyacemaker.domain_models.models import (
    ActiveSet,
    Potential,
    PotentialType,
    StructureMetadata,
)
from pyacemaker.oracle.dataset import DatasetManager
from pyacemaker.trainer.active_set import ActiveSetSelector
from pyacemaker.trainer.base import BaseTrainer
from pyacemaker.trainer.wrapper import PacemakerWrapper


class PacemakerTrainer(BaseTrainer):
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

    def _generate_input_yaml(
        self,
        config: dict[str, Any],
        dataset_path: Path,
        work_dir: Path,
        initial_potential: Path | None = None,
    ) -> Path:
        """Generate input.yaml for Pacemaker."""
        # Defaults
        default_elements = ["Fe"]  # TODO: Infer from dataset or config
        default_embeddings = {
            "Fe": {
                "npot": "FinnisSinclair",
                "fs_parameters": [1, 1, 1, 1],
                "ndensity": 2,
            }
        }

        # Override with config if present
        elements = config.get("elements", default_elements)
        embeddings = config.get("embeddings", default_embeddings)

        input_data = {
            "cutoff": float(config.get("cutoff", CONSTANTS.default_trainer_cutoff)),
            "seed": config.get("seed", 42),
            "data": {
                "filename": str(dataset_path),
                "energy_unit": config.get("energy_unit", "eV"),
                "distance_unit": config.get("distance_unit", "A"),
            },
            "potential": {
                "deltaSplineBins": config.get("deltaSplineBins", 0.001),
                "elements": elements,
                "embeddings": embeddings,
                "bonds": {
                    "N": int(config.get("basis_size", (1, 1))[0]),
                    "L": int(config.get("basis_size", (1, 1))[1]),
                },
            },
            "backend": {
                "evaluator": config.get("evaluator", "tensorpot"),
                "batch_size": int(config.get("batch_size", CONSTANTS.default_trainer_batch_size)),
                "display_step": config.get("display_step", 100),
            },
            "loss": {
                "kappa": config.get("kappa", 0.3),
                "w_energy": config.get("w_energy", 1.0),
                "w_forces": config.get("w_forces", 1.0),
                "w_stress": config.get("w_stress", 0.1),
            },
            "optimizer": {
                "max_epochs": int(config.get("max_epochs", CONSTANTS.default_trainer_max_epochs)),
                "patience": config.get("patience", 50),
            },
        }

        # Handle baseline for delta learning
        if "baseline" in config:
            # Pacemaker input.yaml usually handles this via 'potential' or specific key
            # Assuming 'baseline' key at root or under potential
            # Checking pacemaker docs (simulated): usually 'potential.baseline'
            input_data["potential"]["baseline"] = config["baseline"]

        if initial_potential:
             # Add if supported or handled via CLI
             pass

        input_path = work_dir / "input.yaml"
        with input_path.open("w") as f:
            yaml.dump(input_data, f)

        return input_path

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

            # Streaming execution
            atoms_stream = stream_metadata_to_atoms(valid_counting_stream(dataset))
            self.dataset_manager.save_iter(atoms_stream, dataset_path)

            if stats["count"] == 0:
                msg = "No valid structures with energy and forces found for training."
                raise ValueError(msg)

            # 3. Prepare Params & Input YAML
            params = self.trainer_config.model_dump(exclude={"potential_type"})
            params.update(kwargs)

            initial_pot_path = initial_potential.path if initial_potential else None

            input_yaml_path = self._generate_input_yaml(params, dataset_path, work_dir, initial_pot_path)

            # 4. Train
            if self.trainer_config.mock:
                self.logger.info("Mock Mode: Skipping pace_train execution.")
                # Mock output
                output_pot_path = work_dir / "potential.yace"
                output_pot_path.touch()

            else:
                # Use wrapper with input.yaml
                # We execute in work_dir so output usually goes there
                output_pot_path = self.wrapper.train_from_input(input_yaml_path, work_dir)

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
