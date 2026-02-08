import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

from mlip_autopipec.components.trainer.activeset import ActiveSetSelector
from mlip_autopipec.components.trainer.base import BaseTrainer
from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.domain_models.config import PacemakerInputConfig, PacemakerTrainerConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.utils.security import validate_safe_path

logger = logging.getLogger(__name__)


class PacemakerTrainer(BaseTrainer):
    """
    Pacemaker implementation of the Trainer component.

    This component is responsible for training ACE potentials using the Pacemaker library.
    """

    def __init__(self, config: PacemakerTrainerConfig) -> None:
        super().__init__(config)
        self.config: PacemakerTrainerConfig = config

    @property
    def name(self) -> str:
        return self.config.name

    def train(
        self,
        dataset: Dataset,
        workdir: Path,
        previous_potential: Potential | None = None,
    ) -> Potential:
        """
        Train a potential using the provided dataset.

        Args:
            dataset: The dataset to train on.
            workdir: The directory to store training artifacts.
            previous_potential: Optional previous potential to start training from.

        Returns:
            Potential: The trained potential object.
        """
        workdir.mkdir(parents=True, exist_ok=True)
        safe_workdir = validate_safe_path(workdir)
        logger.info(f"Starting training in {safe_workdir}")

        # 1. Convert Dataset to Pacemaker format
        if self.config.data_format == "extxyz":
            # Override filename extension if extxyz is selected
            # Ensure filename ends with .extxyz
            dataset_filename = Path(self.config.dataset_filename).with_suffix(".extxyz").name
            raw_data_path = safe_workdir / dataset_filename
            dataset.to_extxyz(raw_data_path)
        else:
            raw_data_path = safe_workdir / self.config.dataset_filename
            dataset.to_pacemaker_gzip(raw_data_path)

        training_data_path = raw_data_path

        # 2. Active Set Selection
        if self.config.active_set_selection:
            activeset_path = safe_workdir / self.config.activeset_filename
            logger.info("Running active set selection...")
            selector = ActiveSetSelector(limit=self.config.active_set_limit)
            try:
                training_data_path = selector.select(raw_data_path, activeset_path)
            except Exception as e:
                logger.warning(f"Active set selection failed: {e}. Falling back to full dataset.")
                training_data_path = raw_data_path

        # 3. Generate input.yaml
        input_yaml_path = safe_workdir / self.config.input_filename
        self._generate_input_yaml(input_yaml_path, training_data_path, previous_potential)

        # 4. Run pace_train
        self._run_pace_train(input_yaml_path, safe_workdir)

        # 5. Collect artifacts
        potential_path = safe_workdir / self.config.potential_filename

        if not potential_path.exists():
             # Try to find any yace file
             yace_files = list(workdir.glob("*.yace"))
             # Filter out input potential if it exists
             yace_files = [p for p in yace_files if p.name != "input_potential.yace"]

             if yace_files:
                 # Pick the newest one
                 potential_path = sorted(yace_files, key=lambda p: p.stat().st_mtime)[-1]
             else:
                 msg = f"Training finished but no .yace file found in {workdir}"
                 raise RuntimeError(msg)

        logger.info(f"Training completed. Potential saved to {potential_path}")

        # Parse metrics (placeholder)
        metrics = {"energy_rmse": 0.0, "force_rmse": 0.0}
        metrics_path = workdir / "metrics.json"
        if metrics_path.exists():
             # Load metrics if available
             pass

        return Potential(
            path=potential_path,
            format="yace",
            metrics=metrics,
            creation_date=datetime.now(UTC),
        )

    def _generate_input_yaml(
        self,
        output_path: Path,
        data_path: Path,
        previous_potential: Potential | None
    ) -> None:
        # Physics Baseline injection
        physics_baseline = None
        if self.config.physics_baseline:
            physics_baseline = {
                "type": self.config.physics_baseline.type,
                "params": self.config.physics_baseline.params
            }

        # Initial Potential (Fine-tuning)
        initial_potential_path = None
        if previous_potential:
            # Validate existing potential path
            initial_potential_path = str(validate_safe_path(previous_potential.path, must_exist=True))
        elif self.config.initial_potential:
            initial_potential_path = str(validate_safe_path(Path(self.config.initial_potential), must_exist=True))

        # Basic configuration structure using Pydantic model
        config_model = PacemakerInputConfig(
            cutoff=self.config.cutoff,
            data={
                "filename": data_path.name  # Relative path
            },
            fitting={
                "weight_energy": self.config.fitting_weight_energy,
                "weight_force": self.config.fitting_weight_force,
            },
            backend={
                "evaluator": self.config.backend_evaluator
            },
            b_basis={
                 "max_degree": self.config.basis_size # Rough mapping
            },
            physics_baseline=physics_baseline,
            initial_potential=initial_potential_path
        )

        with output_path.open("w") as f:
            # exclude_none=True to avoid cluttering config with nulls
            yaml.dump(config_model.model_dump(exclude_none=True), f)

    def _run_pace_train(self, input_yaml: Path, workdir: Path) -> None:
        # Assumes pace_train is in PATH
        # Use absolute path for input yaml to be safe
        cmd = ["pace_train", str(input_yaml.resolve())]
        logger.info(f"Executing: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=workdir,
                check=True,
                capture_output=True,
                text=True,
                shell=False,
            )
            # Log stdout/stderr for debugging
            logger.debug(f"pace_train stdout: {result.stdout}")
        except subprocess.CalledProcessError as e:
            msg = f"pace_train failed: {e.stderr}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
