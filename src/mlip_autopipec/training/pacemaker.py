"""
Module for wrapping the Pacemaker training executable.
Manages configuration generation and execution.
"""

import logging
import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

import yaml
from ase import Atoms

from mlip_autopipec.config.schemas.training import TrainingConfig, TrainingResult
from mlip_autopipec.training.metrics import LogParser

logger = logging.getLogger(__name__)

class PacemakerWrapper:
    """
    Wraps Pacemaker training process.
    Handles input generation, execution of the binary, and output verification.

    This class is responsible for the interaction with the external 'pacemaker'
    binary. It generates the necessary 'input.yaml' from the Pydantic configuration,
    launches the subprocess, and parses the result.
    """

    def __init__(self, config: TrainingConfig, work_dir: Path) -> None:
        """
        Initialize the wrapper.

        Args:
            config: Training configuration.
            work_dir: Working directory for training artifacts.
        """
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data_from_stream(self, data_stream: Iterable[Atoms], output_filename: str) -> Path:
        """
        Streams atoms from a generator to a file on disk (extxyz).
        Used to prepare training/test datasets without loading all into memory.
        """
        from ase.io import write
        output_path = self.work_dir / output_filename

        # ase.io.write supports writing multiple images.
        # Ideally we stream-write. 'write' accepts a list, but we can pass an iterator
        # if the format supports it or loop and append.
        # ExtXYZ supports appending.

        if output_path.exists():
            output_path.unlink() # Start fresh

        # Write first frame to initialize file? Or just append all.
        # Actually ase.io.write handles iterables for some formats.
        # Safest for memory is explicit loop if ASE doesn't fully stream write internally.

        try:
            write(str(output_path), data_stream, format="extxyz")
        except Exception as e:
            logger.error(f"Failed to stream write data to {output_path}: {e}")
            raise

        return output_path

    def generate_config(self) -> Path:
        """
        Generates the Pacemaker input YAML configuration file.

        Maps the Pydantic TrainingConfig to the YAML structure required by Pacemaker.

        It strictly validates the generated YAML by parsing it back.

        Returns:
            Path to the generated input.yaml file.
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

        try:
            with config_path.open("w") as f:
                yaml.dump(pacemaker_config, f)

            # Validation: Read back and check structure
            with config_path.open("r") as f:
                loaded_config = yaml.safe_load(f)

            if not isinstance(loaded_config, dict):
                raise ValueError("Generated YAML is not a dictionary.")
            if "cutoff" not in loaded_config:
                raise ValueError("Generated YAML missing required key: cutoff")

        except Exception as e:
            logger.error(f"Failed to generate valid Pacemaker config: {e}")
            raise

        logger.info(f"Generated Pacemaker config at {config_path}")
        return config_path

    def check_output(self, output_path: Path) -> bool:
        """
        Verifies the output potential file exists and is not empty.

        Args:
            output_path: Path to the expected .yace file.

        Returns:
            True if valid, False otherwise.
        """
        return output_path.exists() and output_path.stat().st_size > 0

    def _execute_subprocess(self, cmd: list[str], log_path: Path) -> int:
        """Helper to run the subprocess."""
        try:
            # shell=False prevents shell injection
            # timeout=None by default, but could be added if config supported it
            with log_path.open("w") as log_file:
                result = subprocess.run(
                    cmd,
                    cwd=self.work_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                    shell=False
                )
            return result.returncode
        except subprocess.TimeoutExpired:
            logger.error("Pacemaker execution timed out.")
            return -1
        except subprocess.SubprocessError as e:
            logger.exception(f"Subprocess execution failed: {e}")
            return -1
        except OSError as e:
            logger.exception(f"OS Error during subprocess execution: {e}")
            return -1

    def _resolve_executable(self) -> str:
        """
        Resolves the Pacemaker executable path and performs security checks.
        """
        executable_name = "pacemaker"

        executable = shutil.which(executable_name)
        if not executable:
             # Check if provided as absolute path via env or config (if we had that field)
             # For now, we strictly require it in PATH or standard locations
             raise FileNotFoundError(f"Pacemaker executable '{executable_name}' not found in PATH.")

        # Verify it's an executable file
        p = Path(executable)
        if not (p.exists() and p.is_file() and os.access(p, os.X_OK)):
             raise ValueError(f"Found pacemaker at {executable} but it is not a valid executable.")

        return executable

    def train(self) -> TrainingResult:
        """
        Runs Pacemaker training.

        Executes the 'pacemaker' binary as a subprocess. Captures stdout/stderr
        to a log file and parses metrics upon completion.

        Returns:
            TrainingResult object containing status, metrics, and potential path.
        """
        try:
            config_path = self.generate_config()
            log_path = self.work_dir / "log.txt"

            executable_path = self._resolve_executable()

            # Use full path resolved by shutil.which for security
            cmd = [executable_path, str(config_path.name)]

            logger.info(f"Running Pacemaker: {cmd} in {self.work_dir}")

            returncode = self._execute_subprocess(cmd, log_path)

            if returncode != 0:
                logger.error(f"Pacemaker failed with code {returncode}")
                if log_path.exists():
                    try:
                        content = log_path.read_text()
                        logger.error(f"Pacemaker Output:\n{content}")
                    except Exception:
                        logger.exception("Could not read log file")
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
            logger.error("No valid .yace output found.")
            return TrainingResult(success=False)

        except FileNotFoundError:
            logger.exception("Pacemaker executable not found. Please ensure 'pacemaker' is in PATH.")
            return TrainingResult(success=False)
        except OSError:
            logger.exception("OS Error during training execution")
            return TrainingResult(success=False)
        except Exception:
            logger.exception("Training failed with unexpected error")
            return TrainingResult(success=False)
