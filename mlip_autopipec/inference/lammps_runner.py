import logging
import subprocess
from pathlib import Path

from ase.atoms import Atoms
from ase.io import write

from mlip_autopipec.config.schemas.inference import InferenceConfig, InferenceResult
from mlip_autopipec.inference.inputs import ScriptGenerator

logger = logging.getLogger(__name__)


class LammpsRunner:
    def __init__(self, config: InferenceConfig, work_dir: Path) -> None:
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.generator = ScriptGenerator(config)

    def run(self, atoms: Atoms) -> InferenceResult:
        """
        Executes a LAMMPS simulation for the given atoms object.

        Args:
            atoms: The atomic structure to simulate.

        Returns:
            InferenceResult object containing simulation status and artifacts.
        """
        # Define file paths
        input_file = self.work_dir / "in.lammps"
        data_file = self.work_dir / "data.lammps"
        log_file = self.work_dir / "log.lammps"
        dump_file = self.work_dir / "dump.gamma"

        try:
            # 1. Write Data File
            write(data_file, atoms, format="lammps-data")

            # 2. Generate Input Script
            script_content = self.generator.generate(
                atoms_file=data_file,
                potential_path=self.config.potential_path,
                dump_file=dump_file,
            )

            input_file.write_text(script_content)

            # 3. Execute LAMMPS
            cmd = [
                str(self.config.lammps_executable)
                if self.config.lammps_executable
                else "lmp_serial",
                "-in",
                str(input_file),
                "-log",
                str(log_file),
            ]

            logger.info(f"Starting LAMMPS execution: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, cwd=self.work_dir
            )

            if result.returncode != 0:
                logger.error(f"LAMMPS execution failed with return code {result.returncode}")
                logger.error(f"Stdout: {result.stdout}")
                logger.error(f"Stderr: {result.stderr}")
                success = False
            else:
                logger.info("LAMMPS execution completed successfully.")
                success = True

        except (subprocess.SubprocessError, OSError):
            logger.exception("An error occurred during LAMMPS execution setup or run.")
            # In case of severe failure (e.g. executable not found), we return failure result
            # rather than crashing the pipeline.
            success = False

        # 4. Process Results (Best Effort)
        uncertain_structures = []
        if dump_file.exists() and dump_file.stat().st_size > 0:
            uncertain_structures.append(dump_file)

        # Extract max gamma if log exists (Placeholder for future logic)
        max_gamma = 0.0

        return InferenceResult(
            succeeded=success,
            final_structure=data_file if success else None,
            uncertain_structures=uncertain_structures,
            max_gamma_observed=max_gamma,
        )
