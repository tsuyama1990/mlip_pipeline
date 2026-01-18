"""
Module for wrapping the Pacemaker training executable.
Manages configuration generation, execution, and output parsing.
"""

import gzip
import logging
import pickle
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
        # Ensure work_dir is absolute
        work_dir = work_dir.resolve()
        logger.info(f"Starting training (Generation {generation}) in {work_dir}")
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export Dataset and Identify Elements
        data_path, elements = self._export_and_analyze_dataset(
            config, dataset_builder, work_dir
        )

        # 2. Generate Configuration
        potential_name = f"potential_gen{generation}.yace"
        output_path = work_dir / potential_name
        input_yaml = self._generate_config(
            config_gen, config, data_path, output_path, elements
        )

        # 3. Execute Pacemaker
        stdout, training_time = self._execute_pacemaker(work_dir, input_yaml)

        # 4. Parse Results
        return self._parse_results(
            stdout, work_dir, output_path, training_time, generation
        )

    def _export_and_analyze_dataset(
        self, config: TrainConfig, dataset_builder: DatasetBuilder, work_dir: Path
    ) -> tuple[Path, list[str]]:
        """Exports dataset and extracts unique chemical elements."""
        try:
            data_path = dataset_builder.export(config, work_dir)
        except Exception as e:
            logger.error(f"Failed to export dataset: {e}")
            raise RuntimeError("Dataset export failed") from e

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

        return data_path, sorted_elements

    def _generate_config(
        self,
        config_gen: TrainConfigGenerator,
        config: TrainConfig,
        data_path: Path,
        output_path: Path,
        elements: list[str],
    ) -> Path:
        """Generates the input.yaml configuration file."""
        try:
            input_yaml = config_gen.generate(config, data_path, output_path, elements)
            return input_yaml
        except Exception as e:
            logger.error(f"Failed to generate configuration: {e}")
            raise RuntimeError("Configuration generation failed") from e

    def _execute_pacemaker(self, work_dir: Path, input_yaml: Path) -> tuple[str, float]:
        """Runs the Pacemaker executable and captures output."""
        logger.info(f"Running pacemaker with config: {input_yaml}")
        start_time = time.time()
        # Use filename relative to work_dir if possible, or absolute path
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
            logger.error(f"STDOUT:\n{e.stdout}")
            raise RuntimeError(f"Pacemaker training failed: {e.stderr}") from e
        except FileNotFoundError as e:
            logger.error(f"Pacemaker executable '{self.executable}' not found.")
            raise RuntimeError(
                f"Pacemaker executable '{self.executable}' not found."
            ) from e
        except OSError as e:
            logger.error(f"OS Error executing pacemaker: {e}")
            raise RuntimeError(f"OS Error executing pacemaker: {e}") from e

        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds.")

        return stdout, training_time

    def _parse_results(
        self,
        stdout: str,
        work_dir: Path,
        output_path: Path,
        training_time: float,
        generation: int,
    ) -> TrainingResult:
        """Parses stdout for metrics and verifies output file."""
        re_energy = re.search(r"RMSE\s*\(energy\)\s*:\s*([\d\.eE\-\+]+)", stdout)
        re_forces = re.search(r"RMSE\s*\(forces\)\s*:\s*([\d\.eE\-\+]+)", stdout)

        if not re_energy:
             logger.error("Could not parse RMSE Energy from Pacemaker output.")
             raise RuntimeError("Pacemaker output parsing failed: missing RMSE Energy.")

        if not re_forces:
             logger.error("Could not parse RMSE Forces from Pacemaker output.")
             raise RuntimeError("Pacemaker output parsing failed: missing RMSE Forces.")

        rmse_energy = float(re_energy.group(1))
        rmse_forces = float(re_forces.group(1))

        logger.info(
            f"Final Metrics - RMSE Energy: {rmse_energy}, RMSE Forces: {rmse_forces}"
        )

        if not output_path.exists():
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
