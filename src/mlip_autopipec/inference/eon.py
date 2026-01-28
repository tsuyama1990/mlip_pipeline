import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ase import Atoms
from ase.io import write

from mlip_autopipec.config.schemas.inference import EONConfig
from mlip_autopipec.domain_models.inference_models import InferenceResult
from mlip_autopipec.inference.drivers import pace_driver

logger = logging.getLogger(__name__)


class EONWrapper:
    """
    Wrapper for EON (Kinetic Monte Carlo) software.

    This class manages the configuration, execution, and result parsing of EON simulations
    integrated with the MLIP pipeline via a custom Python driver.
    """

    def __init__(self, config: EONConfig, work_dir: Path) -> None:
        """
        Initialize the EON Wrapper.

        Args:
            config: EON configuration object containing executable path and parameters.
            work_dir: Directory where EON simulation files will be generated and run.
        """
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _write_config(self) -> None:
        """
        Writes the `config.ini` file required by EON.

        The configuration includes job type, temperature, potential name, and any
        additional parameters specified in the `EONConfig`.
        """
        config_path = self.work_dir / "config.ini"
        with config_path.open("w") as f:
            f.write("[Main]\n")
            f.write(f"job = {self.config.job}\n")
            f.write(f"temperature = {self.config.temperature}\n")

            f.write("\n[Potential]\n")
            f.write(f"potential = {self.config.pot_name}\n")

            # Write additional parameters
            if self.config.parameters:
                for k, v in self.config.parameters.items():
                    f.write(f"{k} = {v}\n")

    def _write_pos_con(self, atoms: Atoms) -> None:
        """
        Writes the initial atomic structure to `pos.con` in EON format.

        Args:
            atoms: The ASE Atoms object to write.
        """
        pos_path = self.work_dir / "pos.con"
        write(pos_path, atoms, format="eon")

    def run(self, atoms: Atoms, potential_path: Path, uid: str = "eon_run") -> InferenceResult:
        """
        Orchestrates the execution of an EON simulation.

        Steps:
        1. Writes configuration and structure files.
        2. Generates the wrapper script for the potential driver (`pace_driver.py`).
        3. Executes the EON binary.
        4. Monitors the exit code for Halt signals (Code 100) indicating high uncertainty.

        Args:
            atoms: The starting atomic structure.
            potential_path: Path to the .yace potential file to be used by the driver.
            uid: Unique Identifier for the run.

        Returns:
            InferenceResult: Object containing success status, observed gamma (uncertainty),
            and paths to any uncertain structures found (if halted).
        """
        try:
            self._write_config()
            self._write_pos_con(atoms)

            # Setup Driver
            driver_path = Path(pace_driver.__file__).resolve()
            local_driver = self.work_dir / self.config.pot_name

            # Create a wrapper script to execute the python module
            with local_driver.open("w") as f:
                f.write("#!/bin/bash\n")
                # Ensure we use the same python interpreter
                f.write(f'{sys.executable} {driver_path} "$@"\n')

            # Make executable
            local_driver.chmod(0o755)

            # Prepare Environment
            env = os.environ.copy()
            env["PACE_POTENTIAL_PATH"] = str(potential_path.resolve())
            # Default threshold if not in parameters
            threshold = self.config.parameters.get("uncertainty_threshold", 5.0)
            env["PACE_GAMMA_THRESHOLD"] = str(threshold)

            # Resolve EON Executable
            eon_exe = self.config.eon_executable
            if not eon_exe:
                found = shutil.which("eonclient")
                if found:
                    eon_exe = Path(found)

            if not eon_exe:
                msg = "EON executable not found"
                raise RuntimeError(msg)

            logger.info(f"Starting EON execution: {eon_exe}")

            # Run EON
            result = subprocess.run(
                [str(eon_exe)], cwd=self.work_dir, env=env, capture_output=True, text=True
            )

            logger.debug(f"EON stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"EON stderr: {result.stderr}")

            # Check Exit Codes
            if result.returncode == 0:
                return InferenceResult(
                    uid=uid,
                    succeeded=True,
                    max_gamma_observed=0.0,
                    uncertain_structures=[],
                    halted=False,
                    halt_step=None,
                    error_message=None
                )

            if result.returncode == 100:
                logger.warning("EON halted due to high uncertainty (Exit 100)")

                # Look for bad structure
                uncertain_structures = []
                bad_struct = self.work_dir / "bad_structure.con"
                if bad_struct.exists():
                    uncertain_structures.append(bad_struct)

                return InferenceResult(
                    uid=uid,
                    succeeded=True,  # Halted is a valid "outcome" for AL
                    max_gamma_observed=999.0,  # Indicator
                    uncertain_structures=uncertain_structures,
                    halted=True,
                    halt_step=None,
                    error_message=None
                )

            logger.error(f"EON failed with code {result.returncode}")
            return InferenceResult(
                uid=uid,
                succeeded=False,
                max_gamma_observed=0.0,
                uncertain_structures=[],
                halted=False,
                halt_step=None,
                error_message=f"EON failed with code {result.returncode}"
            )

        except Exception as e:
            logger.exception("EON run failed")
            return InferenceResult(
                uid=uid,
                succeeded=False,
                error_message=str(e),
                max_gamma_observed=0.0,
                uncertain_structures=[],
                halted=False,
                halt_step=None
            )
