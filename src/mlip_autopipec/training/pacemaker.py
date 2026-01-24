"""
Module for wrapping the Pacemaker training executable.
Manages configuration generation and execution.
"""

import logging
import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

import yaml
from ase import Atoms
from ase.io import write

from mlip_autopipec.config.schemas.training import TrainingConfig, TrainingResult
from mlip_autopipec.training.metrics import LogParser
from mlip_autopipec.utils.config_utils import validate_path_safety

logger = logging.getLogger(__name__)


class PacemakerWrapper:
    """
    Wraps Pacemaker training process.
    Handles input generation, execution of the binary, and output verification.
    """

    def __init__(self, config: TrainingConfig, work_dir: Path) -> None:
        """
        Initialize the wrapper.

        Args:
            config: Training configuration.
            work_dir: Working directory for training artifacts.
        """
        self.config = config
        self.work_dir = validate_path_safety(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data_from_stream(self, data_stream: Iterable[Atoms], output_filename: str) -> Path:
        """
        Streams atoms from a generator to a file on disk (extxyz).
        """
        if "/" in output_filename or "\\" in output_filename:
            raise ValueError(f"Invalid filename: {output_filename}")

        output_path = self.work_dir / output_filename

        if output_path.exists():
            output_path.unlink()

        try:
            write(str(output_path), data_stream, format="extxyz")
        except Exception as e:
            logger.error(f"Failed to stream write data to {output_path}: {e}")
            raise

        return output_path

    def _prepare_config(self) -> Path:
        """Helper to generate config."""
        config_path = self.work_dir / "input.yaml"

        validate_path_safety(self.config.training_data_path)
        validate_path_safety(self.config.test_data_path)

        pacemaker_config = {
            "cutoff": self.config.cutoff,
            "data": {
                "filename": self.config.training_data_path,
                "test_filename": self.config.test_data_path,
            },
            "fit": {
                "loss": {"kappa": self.config.kappa, "kappa_f": self.config.kappa_f},
                "optimizer": {
                    "max_iter": self.config.max_num_epochs,
                    "batch_size": self.config.batch_size,
                },
            },
            "b_basis": {"size": self.config.b_basis_size},
            "ladder_step": self.config.ladder_step,
        }

        try:
            with config_path.open("w") as f:
                yaml.dump(pacemaker_config, f)

            # Verify YAML
            with config_path.open("r") as f:
                yaml.safe_load(f)

        except Exception as e:
            logger.error(f"Failed to generate valid Pacemaker config: {e}")
            raise

        logger.info(f"Generated Pacemaker config at {config_path}")
        return config_path

    def generate_config(self) -> Path:
        """Public alias for internal method."""
        return self._prepare_config()

    def check_output(self, output_path: Path) -> bool:
        """Verifies the output potential file exists and is not empty."""
        return output_path.exists() and output_path.stat().st_size > 0

    def _execute_subprocess(self, cmd: list[str], log_path: Path) -> int:
        """Helper to run the subprocess safely."""
        try:
            with log_path.open("w") as log_file:
                # Security: shell=False is crucial here.
                # cmd is a list, arguments are not parsed by shell.
                result = subprocess.run(
                    cmd,
                    cwd=self.work_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                    shell=False,
                )
            return result.returncode

        except subprocess.TimeoutExpired:
            logger.error("Execution timed out.")
            return -1
        except subprocess.SubprocessError as e:
            logger.exception(f"Subprocess execution failed: {e}")
            return -1
        except OSError as e:
            logger.exception(f"OS Error during subprocess execution: {e}")
            return -1

    def _resolve_executable(self, name: str = "pacemaker") -> str:
        """Resolves the executable path and performs security checks."""
        executable = shutil.which(name)
        if not executable:
            raise FileNotFoundError(f"Executable '{name}' not found in PATH.")

        p = Path(executable)
        if not (p.exists() and p.is_file() and os.access(p, os.X_OK)):
            raise ValueError(f"Found {name} at {executable} but it is not a valid executable.")

        return executable

    def train(self, initial_potential: str | Path | None = None) -> TrainingResult:
        """
        Runs Pacemaker training.
        """
        # 1. Validate Inputs
        cmd_extras = []
        if initial_potential:
            safe_pot_path = validate_path_safety(initial_potential)
            if not safe_pot_path.exists():
                raise FileNotFoundError(f"Initial potential not found: {safe_pot_path}")
            cmd_extras.extend(["-p", str(safe_pot_path)])

        try:
            # 2. Prepare Configuration
            config_path = self._prepare_config()
            log_path = self.work_dir / "log.txt"

            # 3. Resolve Executable
            executable_path = self._resolve_executable("pacemaker")

            # 4. Construct Command
            cmd = [executable_path, str(config_path.name)] + cmd_extras

            logger.info(f"Running Pacemaker: {cmd} in {self.work_dir}")

            # 5. Execute
            returncode = self._execute_subprocess(cmd, log_path)

            if returncode != 0:
                logger.error(f"Pacemaker failed with code {returncode}")
                if log_path.exists():
                    try:
                        content = log_path.read_text()
                        logger.error(f"Pacemaker Output:\n{content}")
                    except Exception:
                        pass
                return TrainingResult(success=False)

            # 6. Process Output
            output_yace = self.work_dir / "output.yace"

            if not output_yace.exists():
                yace_files = list(self.work_dir.glob("*.yace"))
                if yace_files:
                    yace_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                    output_yace = yace_files[0]

            if self.check_output(output_yace):
                parser = LogParser()
                metrics = parser.parse_file(log_path)
                return TrainingResult(
                    success=True, potential_path=str(output_yace), metrics=metrics
                )
            logger.error("No valid .yace output found.")
            return TrainingResult(success=False)

        except FileNotFoundError:
            logger.exception("Pacemaker executable not found.")
            return TrainingResult(success=False)
        except Exception:
            logger.exception("Training failed with unexpected error")
            return TrainingResult(success=False)

    def select_active_set(
        self, candidates: list[Atoms], current_potential: str | Path
    ) -> list[int]:
        """
        Selects active set from candidates using the current potential.
        """
        candidates_path = self.work_dir / "candidates.xyz"
        try:
            write(str(candidates_path), candidates, format="extxyz")
        except Exception as e:
            logger.error(f"Failed to write candidates: {e}")
            raise

        try:
            safe_pot_path = validate_path_safety(current_potential)
            if not safe_pot_path.exists():
                raise FileNotFoundError(f"Potential file not found: {safe_pot_path}")

            executable_path = self._resolve_executable("pace_activeset")
            cmd = [executable_path, str(candidates_path), str(safe_pot_path)]

            logger.info(f"Running Active Set Selection: {cmd}")

            result = subprocess.run(
                cmd, cwd=self.work_dir, capture_output=True, text=True, check=False, shell=False
            )

            if result.returncode != 0:
                logger.error(f"Active set selection failed: {result.stderr}")
                raise RuntimeError(f"pace_activeset failed with code {result.returncode}")

            output = result.stdout
            indices = []
            if "Selected indices:" in output:
                part = output.split("Selected indices:")[-1]
                indices = [int(x) for x in re.findall(r"\d+", part)]

            return indices

        except FileNotFoundError:
            logger.error("pace_activeset executable not found.")
            raise
        except Exception:
            logger.exception("Active set selection failed")
            raise
