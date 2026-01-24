import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ase import Atoms
from ase.io import write

from mlip_autopipec.config.schemas.inference import EONConfig
from mlip_autopipec.data_models.inference_models import InferenceResult
from mlip_autopipec.inference.drivers import pace_driver

logger = logging.getLogger(__name__)

class EONWrapper:
    """
    Wrapper for EON (kinetic Monte Carlo).
    """

    def __init__(self, config: EONConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _write_config(self) -> None:
        """Writes config.ini for EON."""
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
                    # If it's a dict/list, we might need specific formatting, but assume scalar
                    f.write(f"{k} = {v}\n")

    def _write_pos_con(self, atoms: Atoms) -> None:
        """Writes initial structure to pos.con."""
        pos_path = self.work_dir / "pos.con"
        # ASE 'eon' format writer
        write(pos_path, atoms, format="eon")

    def run(self, atoms: Atoms, potential_path: Path) -> InferenceResult:
        """
        Runs EON simulation.
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
                f.write(f"{sys.executable} {driver_path} \"$@\"\n")

            # Make executable
            local_driver.chmod(0o755)

            # Prepare Environment
            env = os.environ.copy()
            env["PACE_POTENTIAL_PATH"] = str(potential_path.resolve())
            env["PACE_GAMMA_THRESHOLD"] = str(self.config.parameters.get("uncertainty_threshold", 5.0))

            # Resolve EON Executable
            eon_exe = self.config.eon_executable
            if not eon_exe:
                found = shutil.which("eonclient")
                if found:
                    eon_exe = Path(found)

            if not eon_exe:
                # If mocked in tests, we might proceed, but here we raise
                # We check if we are in a test environment? No, run logic should be strict.
                # However, during tests we might not have eonclient.
                # The test mocks subprocess.run so we assume eon_exe is found or we provide it in config.
                msg = "EON executable not found"
                raise RuntimeError(msg)

            logger.info(f"Starting EON execution: {eon_exe}")

            # Run EON
            result = subprocess.run(
                [str(eon_exe)],
                cwd=self.work_dir,
                env=env,
                capture_output=True,
                text=True
            )

            logger.debug(f"EON stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"EON stderr: {result.stderr}")

            # Check Exit Codes
            if result.returncode == 0:
                return InferenceResult(
                    succeeded=True,
                    max_gamma_observed=0.0,
                    uncertain_structures=[]
                )

            if result.returncode == 100:
                logger.warning("EON halted due to high uncertainty (Exit 100)")

                # Look for bad structure
                uncertain_structures = []
                bad_struct = self.work_dir / "bad_structure.con"
                if bad_struct.exists():
                    uncertain_structures.append(bad_struct)

                return InferenceResult(
                    succeeded=True, # Halted is a valid "outcome" for AL
                    max_gamma_observed=999.0, # Indicator
                    uncertain_structures=uncertain_structures
                )

            logger.error(f"EON failed with code {result.returncode}")
            return InferenceResult(
                succeeded=False,
                max_gamma_observed=0.0,
                uncertain_structures=[]
            )

        except Exception:
            logger.exception("EON run failed")
            return InferenceResult(
                succeeded=False,
                max_gamma_observed=0.0,
                uncertain_structures=[]
            )
