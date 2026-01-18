"""
Module for wrapping the Pacemaker training executable.
Manages configuration generation, execution, and output parsing.
"""

import logging
import re
import subprocess
import time
from pathlib import Path

from mlip_autopipec.config.schemas.training import TrainConfig, TrainingResult
from mlip_autopipec.training.config_gen import TrainConfigGenerator
from mlip_autopipec.training.dataset import DatasetBuilder

logger = logging.getLogger(__name__)


class PacemakerWrapper:
    """
    Wraps Pacemaker training process.

    Attributes:
        executable: Path or name of the pacemaker executable.
    """

    def __init__(self, executable: str = "pacemaker"):
        """
        Initialize the PacemakerWrapper.

        Args:
            executable: Command to run pacemaker (default: "pacemaker").
        """
        self.executable = executable

    def train(
        self,
        config: TrainConfig,
        dataset_builder: DatasetBuilder,
        config_gen: TrainConfigGenerator,
        work_dir: Path,
        generation: int = 0,
    ) -> TrainingResult:
        """
        Executes the training loop.

        Steps:
        1. Export dataset.
        2. Generate config (input.yaml).
        3. Run pacemaker.
        4. Parse logs and result.

        Args:
            config: Training configuration.
            dataset_builder: Instance of DatasetBuilder.
            config_gen: Instance of TrainConfigGenerator.
            work_dir: Working directory for artifacts.
            generation: Current generation index (for active learning).

        Returns:
            TrainingResult object containing metrics and potential path.

        Raises:
            RuntimeError: If training fails or executable is missing.
        """
        logger.info(f"Starting training (Generation {generation}) in {work_dir}")
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export Data
        try:
            data_path = dataset_builder.export(config, work_dir)
        except Exception as e:
            logger.error(f"Failed to export dataset: {e}")
            raise RuntimeError("Dataset export failed") from e

        # Get elements from the dataset for config generation
        # We read the pickle file back. This is efficient enough for typical dataset sizes.
        import gzip
        import pickle

        try:
            with gzip.open(data_path, "rb") as f:
                atoms_list = pickle.load(f)
        except (OSError, pickle.PickleError) as e:
            logger.error(f"Failed to read exported dataset at {data_path}: {e}")
            raise RuntimeError("Could not read exported dataset") from e

        elements = set()
        for at in atoms_list:
            elements.update(at.get_chemical_symbols())
        sorted_elements = sorted(list(elements))
        logger.info(f"Identified elements in training set: {sorted_elements}")

        # 2. Generate Config
        potential_name = f"potential_gen{generation}.yace"
        output_path = work_dir / potential_name

        try:
            input_yaml = config_gen.generate(
                config, data_path, output_path, sorted_elements
            )
        except Exception as e:
            logger.error(f"Failed to generate configuration: {e}")
            raise RuntimeError("Configuration generation failed") from e

        # 3. Run Pacemaker
        logger.info(f"Running pacemaker with config: {input_yaml}")
        start_time = time.time()

        cmd = [self.executable, str(input_yaml.name)]

        try:
            result = subprocess.run(
                cmd, cwd=work_dir, capture_output=True, text=True, check=True
            )
            stdout = result.stdout
            logger.debug(f"Pacemaker STDOUT:\n{stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Pacemaker training failed with return code {e.returncode}.")
            logger.error(f"STDERR:\n{e.stderr}")
            # Log stdout as well for debugging
            logger.error(f"STDOUT:\n{e.stdout}")
            raise RuntimeError(f"Pacemaker training failed: {e.stderr}") from e
        except FileNotFoundError as e:
            logger.error(f"Pacemaker executable '{self.executable}' not found.")
            raise RuntimeError(
                f"Pacemaker executable '{self.executable}' not found."
            ) from e

        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds.")

        # 4. Parse Logs
        rmse_energy = 0.0
        rmse_forces = 0.0

        # Regex parsing for RMSE
        # Adjust patterns as needed for actual Pacemaker version
        re_energy = re.search(r"RMSE\s*\(energy\)\s*:\s*([\d\.eE\-\+]+)", stdout)
        if re_energy:
            rmse_energy = float(re_energy.group(1))

        re_forces = re.search(r"RMSE\s*\(forces\)\s*:\s*([\d\.eE\-\+]+)", stdout)
        if re_forces:
            rmse_forces = float(re_forces.group(1))

        logger.info(f"Final Metrics - RMSE Energy: {rmse_energy}, RMSE Forces: {rmse_forces}")

        # Verify output file exists
        if not output_path.exists():
            # Fallback check for any .yace file
            yace_files = list(work_dir.glob("*.yace"))
            if yace_files:
                logger.warning(
                    f"Expected output {output_path} not found, but found {yace_files[0]}. Using that."
                )
                output_path = yace_files[0]
            else:
                logger.error(f"Output potential file {output_path} not found.")
                raise FileNotFoundError(f"Output potential file {output_path} not found.")

        return TrainingResult(
            potential_path=output_path,
            rmse_energy=rmse_energy,
            rmse_forces=rmse_forces,
            training_time=training_time,
            generation=generation,
        )
