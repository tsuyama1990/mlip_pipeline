import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

from mlip_autopipec.components.trainer.activeset import ActiveSetSelector
from mlip_autopipec.components.trainer.base import BaseTrainer
from mlip_autopipec.constants import (
    PACEMAKER_ACTIVESET_FILENAME,
    PACEMAKER_DATASET_FILENAME,
    PACEMAKER_INPUT_FILENAME,
    PACEMAKER_POTENTIAL_FILENAME,
)
from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.domain_models.config import PacemakerTrainerConfig
from mlip_autopipec.domain_models.potential import Potential

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
        logger.info(f"Starting training in {workdir}")

        # 1. Convert Dataset to Pacemaker format
        raw_data_path = workdir / PACEMAKER_DATASET_FILENAME
        dataset.to_pacemaker_gzip(raw_data_path)

        training_data_path = raw_data_path

        # 2. Active Set Selection
        if self.config.active_set_selection:
            activeset_path = workdir / PACEMAKER_ACTIVESET_FILENAME
            logger.info("Running active set selection...")
            selector = ActiveSetSelector(limit=self.config.active_set_limit)
            try:
                training_data_path = selector.select(raw_data_path, activeset_path)
            except Exception as e:
                logger.warning(f"Active set selection failed: {e}. Falling back to full dataset.")
                training_data_path = raw_data_path

        # 3. Generate input.yaml
        input_yaml_path = workdir / PACEMAKER_INPUT_FILENAME
        self._generate_input_yaml(input_yaml_path, training_data_path, previous_potential)

        # 4. Run pace_train
        self._run_pace_train(input_yaml_path, workdir)

        # 5. Collect artifacts
        potential_path = workdir / PACEMAKER_POTENTIAL_FILENAME

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
        # Basic configuration structure
        config_dict = {
            "cutoff": self.config.cutoff,
            "data": {
                "filename": data_path.name  # Relative path
            },
            "fitting": {
                "weight_energy": self.config.fitting_weight_energy,
                "weight_force": self.config.fitting_weight_force,
            },
            "backend": {
                "evaluator": self.config.backend_evaluator
            },
            "b_basis": {
                 "max_degree": self.config.basis_size # Rough mapping
            }
        }

        # Physics Baseline injection
        if self.config.physics_baseline:
            # Inject baseline configuration
            # This follows the expected structure for delta learning or similar
            config_dict["physics_baseline"] = {
                "type": self.config.physics_baseline.type,
                "params": self.config.physics_baseline.params
            }

        # Initial Potential (Fine-tuning)
        initial_pot = None
        if previous_potential:
            initial_pot = previous_potential.path
        elif self.config.initial_potential:
            initial_pot = Path(self.config.initial_potential)

        if initial_pot:
            # We should probably copy/link the initial potential to workdir
            # or reference it absolutely. Reference absolute is safer.
            config_dict["initial_potential"] = str(initial_pot.resolve())

        with output_path.open("w") as f:
            yaml.dump(config_dict, f)

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
            )  # noqa: S603
            # Log stdout/stderr for debugging
            logger.debug(f"pace_train stdout: {result.stdout}")
        except subprocess.CalledProcessError as e:
            msg = f"pace_train failed: {e.stderr}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
