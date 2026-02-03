import logging
import subprocess
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import write

from mlip_autopipec.config.config_model import LammpsConfig
from mlip_autopipec.domain_models.dynamics import MDResult, MDStatus
from mlip_autopipec.physics.dynamics.input_generator import LammpsInputGenerator
from mlip_autopipec.physics.dynamics.log_parser import LogParser

logger = logging.getLogger(__name__)


class LammpsRunner:
    def __init__(self, config: LammpsConfig) -> None:
        self.config = config

    def run(
        self,
        atoms: Atoms,
        potential_path: Path | None,
        work_dir: Path,
        parameters: dict[str, Any],
    ) -> MDResult:
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. Prepare Input
        # Write data file
        data_file_name = "data.lammps"
        data_file_path = work_dir / data_file_name

        # ASE standard lammps-data writer sorts species alphabetically.
        # We trust ASE and InputGenerator to be consistent.
        write(data_file_path, atoms, format="lammps-data")  # type: ignore[no-untyped-call]

        # Write input file
        input_gen = LammpsInputGenerator(potential_path)
        input_content = input_gen.generate_input(atoms, data_file_name, parameters)

        input_file_path = work_dir / "in.lammps"
        input_file_path.write_text(input_content)

        log_file_path = work_dir / "lammps.log"

        # 2. Execute
        # Parse command string into list for subprocess
        cmd = self.config.command.split()
        cmd.extend(["-in", "in.lammps"])
        cmd.extend(["-log", "lammps.log"])

        logger.info(f"Running LAMMPS in {work_dir}")
        try:
            # We use cwd=work_dir so LAMMPS finds data file and potential easily
            subprocess.run(  # noqa: S603
                cmd,
                cwd=work_dir,
                check=False,  # We handle errors via log parsing
                capture_output=True,
                text=True,
            )
        except Exception:
            logger.exception("Failed to launch LAMMPS process")
            return MDResult(status=MDStatus.FAILED)

        # 3. Parse Log
        parser = LogParser()
        result = parser.parse(log_file_path)

        # Add trajectory path if exists
        traj_path = work_dir / "traj.lammpstrj"
        if traj_path.exists():
            result.trajectory_path = traj_path

        return result
