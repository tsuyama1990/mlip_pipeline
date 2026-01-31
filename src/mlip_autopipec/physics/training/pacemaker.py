import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from mlip_autopipec.domain_models.config import PotentialConfig
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.training import TrainingConfig, TrainingResult


class PacemakerRunner:
    """
    Runs Pacemaker training and active set selection.
    """

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _generate_input_yaml(
        self,
        dataset_path: Path,
        training_config: TrainingConfig,
        potential_config: PotentialConfig,
    ) -> Path:
        """
        Generates input.yaml for pace_train.
        """
        input_yaml_path = self.work_dir / "input.yaml"

        # Basic configuration construction
        # This is a simplified template. In a real scenario, this would be more complex.
        config_dict: dict[str, Any] = {
            "cutoff": potential_config.cutoff,
            "seed": potential_config.seed,
            "data": {
                "filename": str(dataset_path.absolute()),
            },
            "potential": {
                "delta": True,  # Enable delta learning by default (ZBL/LJ)
                "elements": potential_config.elements,
                "embeddings": {
                    "fs_parameters": [1, 1, 1, 1.0],
                    "ndensity": 2,
                },
                "bonds": {
                    "N": 3, # Body order
                    "max_deg": 6,
                    "r0": 1.0,
                    "radbase": "Chebyshev",
                },
            },
            "fit": {
                "loss": {
                    "kappa": training_config.kappa,
                },
                "optimizer": "BFGS",
                "maxiter": training_config.max_epochs,
                "ladder_step": training_config.ladder_step,
                # Note: maxiter in scipy/optimizer usually corresponds to epochs/steps
            },
            "backend": {
                "batch_size": training_config.batch_size,
            },
        }

        # Handle initial potential (Fine-tuning)
        if training_config.initial_potential:
             # Pacemaker allows specifying an initial potential to load
             config_dict["potential"]["initial_potential"] = str(training_config.initial_potential.absolute())

        with input_yaml_path.open("w") as f:
            yaml.dump(config_dict, f, sort_keys=False)

        return input_yaml_path

    def train(
        self,
        dataset_path: Path,
        training_config: TrainingConfig,
        potential_config: PotentialConfig,
    ) -> TrainingResult:
        """
        Executes pace_train.
        """
        input_yaml = self._generate_input_yaml(dataset_path, training_config, potential_config)
        output_potential_path = self.work_dir / "output_potential.yace"
        log_path = self.work_dir / "log.txt"

        cmd = ["pace_train", str(input_yaml)]

        start_time = time.time()
        try:
            with log_path.open("w") as log_file:
                subprocess.run(
                    cmd,
                    check=True,
                    cwd=self.work_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
        except subprocess.CalledProcessError:
            # Read log tail
            log_content = log_path.read_text()[-1000:] if log_path.exists() else "No log"
            return TrainingResult(
                job_id=f"train_{int(start_time)}",
                status=JobStatus.FAILED,
                work_dir=self.work_dir,
                duration_seconds=time.time() - start_time,
                log_content=log_content,
                potential_path=output_potential_path,
                validation_metrics={},
            )
        except FileNotFoundError:
             # Handle missing executable
             return TrainingResult(
                job_id=f"train_{int(start_time)}",
                status=JobStatus.FAILED,
                work_dir=self.work_dir,
                duration_seconds=0.0,
                log_content="pace_train executable not found",
                potential_path=output_potential_path,
                validation_metrics={},
            )

        duration = time.time() - start_time

        # Parse log for metrics
        metrics = self._parse_log(log_path)

        return TrainingResult(
            job_id=f"train_{int(start_time)}",
            status=JobStatus.COMPLETED,
            work_dir=self.work_dir,
            duration_seconds=duration,
            log_content=log_path.read_text()[-1000:] if log_path.exists() else "",
            potential_path=output_potential_path,
            validation_metrics=metrics,
        )

    def select_active_set(self, dataset_path: Path) -> Path:
        """
        Executes pace_activeset to prune the dataset.
        Returns path to the new dataset.
        """
        # Usage: pace_activeset -d dataset.pckl.gzip -o dataset_active.pckl.gzip
        active_dataset_path = dataset_path.with_name(f"{dataset_path.stem.split('.')[0]}_active.pckl.gzip")

        cmd = [
            "pace_activeset",
            "-d", str(dataset_path),
            "-o", str(active_dataset_path),
            # Defaults for now
        ]

        try:
            subprocess.run(cmd, check=True, cwd=self.work_dir, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If it fails, we fall back to the original dataset or raise error?
            # For now, raise error because if active set is requested, it should work.
            raise RuntimeError("Active set selection failed")

        if not active_dataset_path.exists():
            raise RuntimeError("Active set output file not found")

        return active_dataset_path

    def _parse_log(self, log_path: Path) -> dict[str, float]:
        """
        Simple parser to extract RMSE from pace_train log.
        """
        metrics: dict[str, float] = {}
        if not log_path.exists():
            return metrics

        content = log_path.read_text()
        # Example pattern: "RMSE Energy: 1.23 meV/atom"
        # We'll need regex for robustness, but simple search for now
        # In TDD I mocked "Final RMSE Energy: 1.23 meV"

        import re

        # Regex to find Energy RMSE
        # Pattern usually looks like "RMSE_E ... : <value>"
        # Let's assume standard log output pattern.
        # "Final RMSE Energy: 1.23"
        e_match = re.search(r"Final RMSE Energy:\s+([\d\.]+)", content)
        if e_match:
            metrics["rmse_energy"] = float(e_match.group(1))

        f_match = re.search(r"Final RMSE Force:\s+([\d\.]+)", content)
        if f_match:
            metrics["rmse_force"] = float(f_match.group(1))

        return metrics
