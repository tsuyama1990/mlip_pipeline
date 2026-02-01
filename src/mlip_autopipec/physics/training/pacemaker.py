import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from mlip_autopipec.domain_models.config import PotentialConfig
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.training import TrainingConfig, TrainingResult

logger = logging.getLogger("mlip_autopipec")


class PacemakerRunner:
    """
    Runs the Pacemaker training pipeline: Active Set Selection -> Training.
    """

    def __init__(
        self,
        work_dir: Path,
        train_config: TrainingConfig,
        potential_config: PotentialConfig,
    ):
        self.work_dir = work_dir
        self.train_config = train_config # Use train_config to match manager usage
        # Also alias to config to keep internal usage consistent if needed, but standardizing on train_config is better.
        # However, to avoid breaking existing code in this file, I'll alias self.config = train_config
        self.config = train_config
        self.pot_config = potential_config
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def train(self, dataset_path: Path) -> TrainingResult:
        """
        Execute the training process.

        Args:
            dataset_path: Path to the dataset file (can be extxyz or pckl.gzip).
                          If extxyz, active set selection (if enabled) will produce pckl.gzip.
        """
        logger.info(f"Starting Pacemaker training in {self.work_dir}")

        # 1. Active Set Selection (Optional but recommended)
        final_dataset_path = dataset_path

        if self.config.active_set_optimization:
            logger.info("Running active set selection...")
            try:
                active_set_path = self.select_active_set(dataset_path)
                logger.info(f"Active set selected: {active_set_path}")
                final_dataset_path = active_set_path
            except subprocess.CalledProcessError as e:
                logger.warning(f"Active set selection failed, falling back to full dataset: {e}")
                pass

        # 2. Generate input.yaml
        input_yaml_path = self.work_dir / "input.yaml"
        self._generate_input_yaml(final_dataset_path, input_yaml_path)

        # 3. Run pace_train
        log_path = self.work_dir / "log.txt"
        cmd = ["pace_train", str(input_yaml_path)]

        if self.config.initial_potential:
            cmd.extend(["--initial_potential", str(self.config.initial_potential)])

        start_time = datetime.now()
        status = JobStatus.PENDING

        try:
            with open(log_path, "w") as f:
                subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
            status = JobStatus.COMPLETED
        except subprocess.CalledProcessError as e:
            status = JobStatus.FAILED
            logger.error(f"pace_train failed: {e}")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 4. Parse log
        log_content = log_path.read_text() if log_path.exists() else ""
        metrics = self._parse_log(log_content)

        # 5. Locate output potential
        potential_path = self.work_dir / "potential.yace"

        if status == JobStatus.COMPLETED and not potential_path.exists():
            logger.error("Training completed but potential file not found.")
            status = JobStatus.FAILED

        return TrainingResult(
            job_id=self.work_dir.name,
            status=status,
            work_dir=self.work_dir,
            duration_seconds=duration,
            log_content=log_content,
            potential_path=potential_path,
            validation_metrics=metrics,
        )

    def select_active_set(self, dataset_path: Path) -> Path:
        """
        Run pace_activeset to reduce the dataset.
        """
        output_path = self.work_dir / "train_active.pckl.gzip"

        cmd = ["pace_activeset", str(dataset_path), "--output", str(output_path)]

        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    def _generate_input_yaml(self, dataset_path: Path, output_path: Path) -> None:
        """
        Generate input.yaml for pace_train.
        """
        # Configuration from PotentialConfig
        data: Dict[str, Any] = {
            "cutoff": self.pot_config.cutoff,
            "seed": self.pot_config.seed,
            "data": {
                "filename": str(dataset_path.absolute())
            },
            "potential": {
                "elements": self.pot_config.elements,
                "deltaSplineBins": self.pot_config.delta_spline_bins,
                "embeddings": {
                    "ALL": {
                        "npot": self.pot_config.ace_params.npot,
                        "fs_parameters": self.pot_config.ace_params.fs_parameters,
                        "ndensity": self.pot_config.ace_params.ndensity
                    }
                },
                "bonds": {
                    "ALL": {
                        "dmin": 0,
                        "dmax": self.pot_config.cutoff
                    }
                }
            },
            "fit": {
                "loss": {
                    "kappa": self.config.kappa,
                    "w_energy": self.pot_config.loss_weight_energy,
                    "w_forces": self.pot_config.loss_weight_forces,
                    "w_stress": self.pot_config.loss_weight_stress,
                },
                "optimizer": {
                    "max_epochs": self.config.max_epochs,
                    "batch_size": self.config.batch_size,
                    "ladder_step": self.config.ladder_step,
                }
            },
            "backend": {
                "evaluator": "tensorpot"
            },
            "output": {
                "potential": "potential.yace"
            }
        }

        # Add Repulsion if Hybrid
        if self.pot_config.pair_style == "hybrid/overlay":
            data["potential"]["repulsion"] = {
                "type": "zbl",
                "inner_cutoff": self.pot_config.zbl_inner_cutoff,
                "outer_cutoff": self.pot_config.zbl_outer_cutoff
            }

        with open(output_path, "w") as f:
            yaml.dump(data, f)

    def _parse_log(self, content: str) -> Dict[str, float]:
        """
        Parse training log for metrics.
        """
        metrics = {}
        for line in content.splitlines():
            lower_line = line.lower()
            if "rmse energy" in lower_line:
                try:
                    val = float(line.split(":")[-1].strip())
                    metrics["energy"] = val
                except ValueError:
                    pass
            elif "rmse force" in lower_line:
                try:
                    val = float(line.split(":")[-1].strip())
                    metrics["force"] = val
                except ValueError:
                    pass
        return metrics
