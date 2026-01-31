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
    Runs Pacemaker commands for training and active set selection.
    """

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        dataset_path: Path,
        training_config: TrainingConfig,
        potential_config: PotentialConfig,
    ) -> TrainingResult:
        """
        Run the training process.
        """
        start_time = time.time()

        # Prepare input.yaml
        input_yaml_path = self.work_dir / "input.yaml"
        input_data = self._generate_input_yaml(
            dataset_path, training_config, potential_config
        )

        with open(input_yaml_path, "w") as f:
            yaml.dump(input_data, f, sort_keys=False)

        # Run pace_train
        log_path = self.work_dir / "log.txt"
        cmd = ["pace_train", str(input_yaml_path)]

        try:
            with open(log_path, "w") as log_file:
                subprocess.run(
                    cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT
                )
            status = JobStatus.COMPLETED
        except subprocess.CalledProcessError:
            status = JobStatus.FAILED
            # Even if failed, we proceed to return result with FAILED status?
            # Or raise? The job model suggests we return a result.
            # But usually we might want to raise if it's catastrophic.
            # Let's verify if 'potential.yace' exists.

        duration = time.time() - start_time
        log_content = log_path.read_text() if log_path.exists() else "No log"

        potential_path = self.work_dir / "potential.yace"

        metrics = {}
        if status == JobStatus.COMPLETED and potential_path.exists():
            metrics = self._parse_metrics(log_content)
        else:
            status = JobStatus.FAILED

        return TrainingResult(
            job_id=f"train_{int(start_time)}",
            status=status,
            work_dir=self.work_dir,
            duration_seconds=duration,
            log_content=log_content[-1000:],  # Tail
            potential_path=potential_path,
            validation_metrics=metrics,
        )

    def select_active_set(
        self,
        dataset_path: Path,
        training_config: TrainingConfig,
        potential_config: PotentialConfig,
    ) -> Path:
        """
        Run active set selection.
        """
        # Output path
        output_dataset = self.work_dir / "train_active.pckl.gzip"

        # We need an input.yaml for activeset too? Usually pace_activeset takes args or yaml.
        # pace_activeset -d <data> -o <output> ...
        # But it also needs basis set definition (cutoff etc) which are in input.yaml.
        # Let's assume we can reuse input.yaml or pass args.
        # For simplicity, let's create a minimal input.yaml if needed, or just pass args if supported.
        # According to spec: "Run pace_activeset."

        # Let's assume we use the same generated input.yaml but override data/output?
        # Or standard usage: pace_activeset input.yaml -d data -o output

        input_yaml_path = self.work_dir / "input.yaml"
        if not input_yaml_path.exists():
            input_data = self._generate_input_yaml(
                dataset_path, training_config, potential_config
            )
            with open(input_yaml_path, "w") as f:
                yaml.dump(input_data, f, sort_keys=False)

        cmd = [
            "pace_activeset",
            str(input_yaml_path),
            "--dataset",
            str(dataset_path),
            "--output",
            str(output_dataset),
        ]

        log_path = self.work_dir / "activeset.log"
        try:
            with open(log_path, "w") as log_file:
                subprocess.run(
                    cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT
                )
        except subprocess.CalledProcessError as e:
            msg = f"pace_activeset failed. Check {log_path}"
            raise RuntimeError(msg) from e

        return output_dataset

    def _generate_input_yaml(
        self,
        dataset_path: Path,
        training_config: TrainingConfig,
        potential_config: PotentialConfig,
    ) -> dict[str, Any]:
        """Generate dictionary for input.yaml"""

        # Basic structure
        return {
            "cutoff": potential_config.cutoff,
            "seed": potential_config.seed,
            "elements": potential_config.elements,
            "data": {
                "filename": str(dataset_path),
                "energy_key": "energy",
                "forces_key": "forces",
                "stress_key": "stress",  # Optional
            },
            "potential": {
                "delta": {
                    "type": "ZBL",  # Hardcoded for now as per spec suggestion
                    "inner_cutoff": 0.1,
                    "outer_cutoff": 2.0,  # Heuristic
                },
                "bonds": {
                    elem: {
                        "element": elem,
                        "r0": 1.0,  # Heuristic, maybe should be in config
                    }
                    for elem in potential_config.elements
                },
            },
            "fit": {
                "loss": {
                    "kappa": training_config.kappa,
                    "L1_coeffs": 1e-8,
                    "L2_coeffs": 1e-8,
                },
                "optimizer": {
                    "max_epochs": training_config.max_epochs,
                    "batch_size": training_config.batch_size,
                    "ladder_step": training_config.ladder_step,
                },
            },
            "backend": {
                "evaluator": "tensorpot",
                "batch_size": training_config.batch_size,
                "display_step": 20,
            },
        }

    def _parse_metrics(self, log_content: str) -> dict[str, float]:
        """Parse RMSE from log."""
        metrics = {}
        # Simple parsing logic
        # Example log: "RMSE Energy: 0.005"
        for line in log_content.splitlines():
            if "RMSE Energy" in line:
                try:
                    metrics["rmse_energy"] = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            if "RMSE Force" in line:
                try:
                    metrics["rmse_force"] = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
        return metrics
