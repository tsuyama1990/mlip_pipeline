"""
Module for wrapping the Pacemaker training executable.
Manages configuration generation and execution.
"""

import logging
import subprocess
import yaml
from pathlib import Path

from mlip_autopipec.config.schemas.training import TrainingConfig, TrainingResult
from mlip_autopipec.training.metrics import LogParser

logger = logging.getLogger(__name__)

class PacemakerWrapper:
    """
    Wraps Pacemaker training process.
    """

    def __init__(self, config: TrainingConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def generate_config(self) -> Path:
        """
        Generates the Pacemaker input YAML.
        """
        config_path = self.work_dir / "input.yaml"

        pacemaker_config = {
            "cutoff": self.config.cutoff,
            "data": {
                "filename": self.config.training_data_path,
                "test_filename": self.config.test_data_path
            },
            "fit": {
                "loss": {
                    "kappa": self.config.kappa,
                    "kappa_f": self.config.kappa_f
                },
                "optimizer": {
                    "max_iter": self.config.max_iter
                }
            },
            "b_basis": {
                "size": self.config.b_basis_size
            }
        }

        with open(config_path, "w") as f:
            yaml.dump(pacemaker_config, f)

        logger.info(f"Generated Pacemaker config at {config_path}")
        return config_path

    def check_output(self, output_path: Path) -> bool:
        """Verifies the output potential file."""
        return output_path.exists() and output_path.stat().st_size > 0

    def train(self) -> TrainingResult:
        """
        Runs Pacemaker training.
        """
        config_path = self.generate_config()
        log_path = self.work_dir / "log.txt"

        cmd = ["pacemaker", str(config_path.name)]

        logger.info(f"Running Pacemaker in {self.work_dir}")
        try:
            with open(log_path, "w") as log_file:
                result = subprocess.run(
                    cmd,
                    cwd=self.work_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False
                )

            if result.returncode != 0:
                logger.error(f"Pacemaker failed with code {result.returncode}")
                return TrainingResult(success=False)

            output_yace = self.work_dir / "output.yace"

            if not output_yace.exists():
                 yace_files = list(self.work_dir.glob("*.yace"))
                 if yace_files:
                     output_yace = yace_files[0]

            if self.check_output(output_yace):
                 parser = LogParser()
                 metrics = parser.parse_file(log_path)

                 return TrainingResult(
                     success=True,
                     potential_path=str(output_yace),
                     metrics=metrics
                 )
            else:
                 logger.error("No valid .yace output found.")
                 return TrainingResult(success=False)

        except Exception as e:
            logger.exception(f"Training failed: {e}")
            return TrainingResult(success=False)
