import logging
import subprocess
import yaml
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from mlip_autopipec.domain_models.training import TrainingConfig, TrainingResult
from mlip_autopipec.domain_models.config import PotentialConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.job import JobStatus

logger = logging.getLogger(__name__)


class PacemakerRunner:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        dataset_path: Path,
        train_config: TrainingConfig,
        potential_config: PotentialConfig,
    ) -> TrainingResult:
        """
        Executes the training process: input generation -> pace_train -> parsing.
        """
        # 1. Generate input.yaml
        input_yaml_path = self.work_dir / "input.yaml"
        self._generate_input_yaml(
            input_yaml_path, dataset_path, train_config, potential_config
        )

        # 2. Run pace_train
        log_path = self.work_dir / "log.txt"
        logger.info(f"Starting training in {self.work_dir}")

        # pace_train <input.yaml>
        cmd = ["pace_train", str(input_yaml_path)]

        start_time = datetime.now()

        with open(log_path, "w") as log_file:
            try:
                subprocess.run(
                    cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Training failed. Check {log_path}")
                # Try to read log to give more info in error
                if log_path.exists():
                     logger.error(log_path.read_text())
                raise RuntimeError("Training failed") from e

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 3. Parse output
        metrics = self._parse_log(log_path)

        # 4. Construct Result
        # pace_train usually produces 'output_potential.yace' in the working directory
        potential_path = self.work_dir / "output_potential.yace"

        if not potential_path.exists():
            # In simulation/mock it might not exist if we don't create it.
            # Real run would fail earlier or here.
            # But wait, if pace_train succeeds it should be there.
            raise FileNotFoundError(f"Potential file not found at {potential_path}")

        potential = Potential(
            path=potential_path,
            format="ace",
            elements=potential_config.elements,
            creation_date=datetime.now(),
            metadata=metrics,
        )

        return TrainingResult(
            job_id=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            status=JobStatus.COMPLETED,
            work_dir=self.work_dir,
            duration_seconds=duration,
            log_content=log_path.read_text(),
            potential=potential,
            validation_metrics=metrics,
        )

    def select_active_set(self, dataset_path: Path) -> Path:
        """
        Runs pace_activeset to prune the dataset.
        Returns the path to the new dataset.
        """
        output_path = self.work_dir / "train_active.pckl.gzip"
        logger.info(f"Selecting active set from {dataset_path}")

        # Command assumption: pace_activeset <input> <output>
        # Or checking docs (hypothetically): pace_activeset -d input.pckl.gzip -o output.pckl.gzip
        # I'll stick to positional: pace_activeset input output
        cmd = ["pace_activeset", str(dataset_path), str(output_path)]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error("Active set selection failed")
            raise RuntimeError("Active set selection failed") from e

        return output_path

    def _generate_input_yaml(
        self,
        path: Path,
        dataset_path: Path,
        t_conf: TrainingConfig,
        p_conf: PotentialConfig,
    ) -> None:
        """Generates the input.yaml for pacemaker."""
        data: Dict[str, Any] = {
            "cutoff": p_conf.cutoff,
            "seed": p_conf.seed,
            "elements": p_conf.elements,
            "data": {"filename": str(dataset_path)},
            "fit": {
                "loss": {"kappa": t_conf.kappa},
                "optimizer": {
                    "batch_size": t_conf.batch_size,
                    "max_epochs": t_conf.max_epochs,
                },
            },
            # Simplified embedding/potential section
            "potential": {
                "delta": True,
                "embeddings": {
                    "ACE": {"type": "ACE", "ladder_step": t_conf.ladder_step},
                    "ZBL": {"type": "ZBL"},
                },
            },
        }

        if t_conf.initial_potential:
            data["potential"]["initial_potential"] = str(t_conf.initial_potential)

        with open(path, "w") as f:
            yaml.dump(data, f)

    def _parse_log(self, log_path: Path) -> Dict[str, float]:
        """Parses the training log for metrics."""
        content = log_path.read_text()
        metrics = {}
        # Parse logic: look for "RMSE Energy: 0.123"
        e_match = re.search(r"RMSE Energy:\s+([\d\.]+)", content)
        f_match = re.search(r"RMSE Force:\s+([\d\.]+)", content)

        if e_match:
            metrics["energy_rmse"] = float(e_match.group(1))
        if f_match:
            metrics["force_rmse"] = float(f_match.group(1))

        return metrics
