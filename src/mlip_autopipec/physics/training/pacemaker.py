import subprocess
import logging
import re
from pathlib import Path

from mlip_autopipec.domain_models.training import TrainingConfig, TrainingResult
from mlip_autopipec.domain_models.config import PotentialConfig
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.infrastructure import io

logger = logging.getLogger("mlip_autopipec")


class PacemakerRunner:
    """
    Runs Pacemaker training tasks (pace_train, pace_activeset).
    """

    def __init__(self, config: TrainingConfig, potential_config: PotentialConfig, work_dir: Path):
        self.config = config
        self.potential_config = potential_config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def select_active_set(self, dataset_path: Path) -> Path:
        """
        Run pace_activeset to select sparse subset of structures.
        Returns path to the new dataset.

        WARNING: This operation invokes `pace_activeset` which may load the entire
        dataset into memory. Ensure sufficient RAM is available for large datasets.
        """
        if not self.config.active_set_optimization:
            logger.info("Active set optimization disabled.")
            return dataset_path

        # Check dataset size to warn about potential OOM
        try:
            file_size_mb = dataset_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 500: # Warn if > 500MB
                 logger.warning(
                     f"Dataset size is {file_size_mb:.2f} MB. "
                     "pace_activeset might consume significant memory."
                 )
        except Exception:
            pass

        output_dataset = self.work_dir / "train_active.pckl.gzip"
        cmd = [
            "pace_activeset",
            "--data-config", f"filename={dataset_path}",
            "--output", str(output_dataset),
        ]

        logger.info(f"Running pace_activeset: {' '.join(cmd)}")
        try:
            subprocess.run(
                cmd,
                cwd=self.work_dir,
                check=True,
                capture_output=True,
                text=True
            )
            return output_dataset
        except subprocess.CalledProcessError as e:
            logger.error(f"pace_activeset failed: {e.stderr}")
            raise RuntimeError(f"pace_activeset failed: {e.stderr}") from e

    def train(self, dataset_path: Path) -> TrainingResult:
        """
        Run pace_train.
        """
        # 1. Generate input.yaml
        input_yaml_path = self.work_dir / "input.yaml"
        self._generate_input_yaml(dataset_path, input_yaml_path)

        # 2. Execute
        cmd = ["pace_train", str(input_yaml_path)]
        log_path = self.work_dir / "training.log"

        logger.info(f"Running pace_train: {' '.join(cmd)}")

        import time
        start_time = time.time()

        try:
            with open(log_path, "w") as f_log:
                # We don't assign the result to a variable since we check exception
                subprocess.run(
                    cmd,
                    cwd=self.work_dir,
                    check=True,
                    stdout=f_log,
                    stderr=subprocess.STDOUT,
                    text=True
                )

            status = JobStatus.COMPLETED
            log_content = log_path.read_text()[-2000:]

            potential_path = self.work_dir / "output_potential.yace"
            metrics = self._parse_metrics(log_path)

        except subprocess.CalledProcessError as e:
            status = JobStatus.FAILED
            if log_path.exists():
                log_content = log_path.read_text()[-2000:]
            else:
                log_content = str(e)
            potential_path = Path("failed")
            metrics = {}

        duration = time.time() - start_time

        return TrainingResult(
            job_id="train_" + self.work_dir.name,
            status=status,
            work_dir=self.work_dir,
            duration_seconds=duration,
            log_content=log_content,
            potential_path=potential_path,
            validation_metrics=metrics
        )

    def _generate_input_yaml(self, dataset_path: Path, output_path: Path) -> None:
        """
        Construct input.yaml for pacemaker.
        """
        config_dict = {
            "cutoff": self.potential_config.cutoff,
            "data": {
                "filename": str(dataset_path.resolve())
            },
            "bonds": {
                "element": self.potential_config.elements,
            },
            "fit": {
                "loss": {
                    "kappa": self.config.kappa
                },
                "optimizer": "BFGS"
            },
            "backend": {
                "evaluator": "tensorpot",
                "batch_size": self.config.batch_size,
                "display_step": 50
            },
            "ladder_step": self.config.ladder_step,
            "max_num_epochs": self.config.max_epochs,
            "YAML": "output_potential.yace"
        }

        io.dump_yaml(config_dict, output_path)

    def _parse_metrics(self, log_path: Path) -> dict[str, float]:
        """
        Parse RMSE from log file.
        """
        content = log_path.read_text()
        metrics = {}

        e_match = re.search(r"RMSE Energy.*:\s+([\d\.]+)", content, re.IGNORECASE)
        if e_match:
            metrics["rmse_energy"] = float(e_match.group(1))

        f_match = re.search(r"RMSE Force.*:\s+([\d\.]+)", content, re.IGNORECASE)
        if f_match:
            metrics["rmse_force"] = float(f_match.group(1))

        return metrics
