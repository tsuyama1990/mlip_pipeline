import logging
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

import yaml

from mlip_autopipec.domain_models.config import TrainerConfig
from mlip_autopipec.domain_models.datastructures import Potential, Structure
from mlip_autopipec.domain_models.enums import ActiveSetMethod
from mlip_autopipec.trainer.dataset_manager import DatasetManager
from mlip_autopipec.trainer.delta_learning import DeltaLearning
from mlip_autopipec.trainer.interface import BaseTrainer

logger = logging.getLogger(__name__)


class PacemakerTrainer(BaseTrainer):
    """Trainer implementation using Pacemaker."""

    def __init__(self, work_dir: Path, config: TrainerConfig) -> None:
        self.work_dir = work_dir
        self.config = config
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_manager = DatasetManager(self.work_dir)

    def train(self, structures: Iterable[Structure]) -> Potential:
        """
        Trains a potential on the given structures using Pacemaker.

        Args:
            structures: An iterable of labeled structures.

        Returns:
            A Potential object representing the trained model.
        """
        logger.info("PacemakerTrainer: Starting training process...")

        # 1. Collect structures into a list
        structure_list = list(structures)
        if not structure_list:
            msg = "No structures provided for training."
            logger.error(msg)
            raise ValueError(msg)

        # 2. Create Dataset
        dataset_path = self.work_dir / "dataset.pckl.gzip"
        self.dataset_manager.create_dataset(structure_list, dataset_path)
        logger.info(f"Dataset created at {dataset_path}")

        # 3. Active Set Selection (if enabled)
        training_dataset_path = dataset_path
        if self.config.active_set_method != ActiveSetMethod.NONE:
            count = int(len(structure_list) * self.config.selection_ratio)
            if count > 0:
                logger.info(f"Selecting active set of size {count} using {self.config.active_set_method}")
                training_dataset_path = self.dataset_manager.select_active_set(dataset_path, count)
            else:
                logger.warning("Active set count is 0, skipping active set selection.")

        # 4. Generate Input YAML
        input_yaml_path = self.generate_input_yaml(structure_list, training_dataset_path)

        # 5. Run Training
        # Run in work_dir so output potential is generated there
        cmd = ["pace_train", input_yaml_path.name]
        logger.info(f"Running pace_train: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=self.work_dir)  # noqa: S603

        if result.returncode != 0:
            msg = f"pace_train failed: {result.stderr}"
            logger.error(msg)
            raise RuntimeError(msg)

        # 6. Verify Output
        potential_path = self.work_dir / "output_potential.yace"
        if not potential_path.exists():
            # Try to find any .yace file in work_dir if default name differs
            yace_files = list(self.work_dir.glob("*.yace"))
            if yace_files:
                potential_path = yace_files[0]
            else:
                msg = f"pace_train did not produce a .yace file in {self.work_dir}"
                logger.error(msg)
                raise FileNotFoundError(msg)

        return Potential(
            path=potential_path,
            format="yace",
            parameters=self.config.model_dump(),
        )

    def generate_input_yaml(self, structures: list[Structure], dataset_path: Path) -> Path:
        """Generates the input.yaml configuration for Pacemaker."""
        input_yaml_path = self.work_dir / "input.yaml"

        # Determine elements from structures
        elements = sorted({atom.symbol for s in structures for atom in s.ase_atoms})

        # Delta Learning config string
        delta_config_str = DeltaLearning.get_config(elements, self.config.delta_learning)

        # Basic Pacemaker configuration structure
        config_dict = {
            "cutoff": self.config.cutoff,
            "seed": 42,
            "b_basis": {
                "max_deg": self.config.order,
                "n_basis": self.config.basis_size,
                "species": elements,
            },
            "fit": {
                "loss": {"kappa": 0.01, "L1_coeffs": 1e-8, "L2_coeffs": 1e-8},
                "optimizer": {
                    "max_epochs": self.config.max_epochs,
                    "batch_size": self.config.batch_size,
                },
            },
            "data": {
                "filename": str(dataset_path.absolute())
            },
        }

        yaml_str = yaml.dump(config_dict)
        full_config = yaml_str + "\n" + delta_config_str

        input_yaml_path.write_text(full_config)
        return input_yaml_path

    def select_active_set(self, structures: Iterable[Structure], count: int) -> Iterator[Structure]:
        """
        Selects structures using DatasetManager.

        Warning: This implementation falls back to returning the first `count` structures
        if `pyace` is not available to read back the selected dataset.
        """
        structure_list = list(structures)
        if not structure_list:
            return

        # Create temporary dataset
        dataset_path = self.work_dir / "temp_candidates.pckl.gzip"
        self.dataset_manager.create_dataset(structure_list, dataset_path)

        # TODO: Implement reading back from active_set_path using pyace if available.
        # For now, we fallback to passing through the first `count` structures
        # to ensure the pipeline continues even without pyace installed.

        logger.warning(
            "PacemakerTrainer.select_active_set: Reading back from Pacemaker dataset is not implemented "
            "(requires pyace). Returning first N structures."
        )

        for i, s in enumerate(structure_list):
            if i < count:
                yield s
