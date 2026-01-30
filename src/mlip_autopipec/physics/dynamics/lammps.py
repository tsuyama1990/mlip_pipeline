import logging
from pathlib import Path
from typing import Optional
import time
import subprocess
import uuid
import datetime

import ase.io

from mlip_autopipec.domain_models.config import LammpsConfig, MDParams, PotentialConfig
from mlip_autopipec.domain_models.job import JobStatus, LammpsResult
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.infrastructure import io

logger = logging.getLogger(__name__)


class LammpsRunner:
    """
    Executes LAMMPS simulations.
    """

    def __init__(self, config: LammpsConfig, base_work_dir: Path = Path("_work_md")):
        self.config = config
        self.base_work_dir = base_work_dir

    def run(self, structure: Structure, params: MDParams, potential_config: PotentialConfig) -> LammpsResult:
        """
        Run a LAMMPS simulation.

        Args:
            structure: Initial atomic structure.
            params: MD simulation parameters.
            potential_config: Configuration for the interatomic potential.

        Returns:
            LammpsResult object.
        """
        start_time = time.time()

        # Create a persistent directory
        job_id = f"job_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        work_dir = self.base_work_dir / job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running LAMMPS in {work_dir}")

        try:
            # 1. Write inputs
            self._write_inputs(work_dir, structure, params, potential_config)

            # 2. Run LAMMPS
            log_content = self._execute_lammps(work_dir)

            # 3. Parse output
            final_structure, trajectory_path = self._parse_output(work_dir)

            duration = time.time() - start_time

            return LammpsResult(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                work_dir=work_dir,
                duration_seconds=duration,
                log_content=log_content,
                final_structure=final_structure,
                trajectory_path=trajectory_path
            )

        except subprocess.TimeoutExpired:
            logger.error("LAMMPS simulation timed out.")
            return LammpsResult(
                job_id=job_id,
                status=JobStatus.TIMEOUT,
                work_dir=work_dir,
                duration_seconds=time.time() - start_time,
                log_content="Timeout"
            )
        except Exception as e:
            logger.exception("LAMMPS simulation failed.")
            return LammpsResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                work_dir=work_dir,
                duration_seconds=time.time() - start_time,
                log_content=str(e)
            )

    def _write_inputs(self, work_dir: Path, structure: Structure, params: MDParams, potential_config: PotentialConfig) -> None:
        """Write data.lammps and in.lammps."""
        # Write data file
        atoms = structure.to_ase()
        # ASE writes lammps-data
        ase.io.write(work_dir / "data.lammps", atoms, format="lammps-data") # type: ignore[no-untyped-call]

        # Generate Pair Coeff lines
        pair_coeff_lines = "\n".join([f"pair_coeff      {line}" for line in potential_config.pair_coeff])

        # Write input script
        input_script = f"""
units           metal
atom_style      atomic
boundary        p p p

read_data       data.lammps

pair_style      {potential_config.pair_style}
{pair_coeff_lines}

neighbor        0.3 bin
neigh_modify    delay 0 every 20 check no

thermo          100
thermo_style    custom step temp pe ke etotal press

# NVT
velocity        all create {params.temperature} 12345 dist gaussian
fix             1 all nvt temp {params.temperature} {params.temperature} $(100.0*dt)

timestep        {params.timestep}

dump            1 all custom 100 dump.lammpstrj id type x y z
dump_modify     1 sort id

run             {params.n_steps}
"""
        (work_dir / "in.lammps").write_text(input_script)

    def _execute_lammps(self, work_dir: Path) -> str:
        """Run the LAMMPS process."""
        cmd_str = self.config.command
        cmd_list = cmd_str.split()

        # Add input file argument
        cmd_list.extend(["-in", "in.lammps"])

        return_code, stdout, stderr = io.run_subprocess(
            cmd_list,
            cwd=work_dir,
            timeout=self.config.timeout,
            env=None
        )

        if return_code != 0:
            msg = f"LAMMPS exited with code {return_code}. Stderr: {stderr}"
            raise RuntimeError(msg)

        return stdout

    def _parse_output(self, work_dir: Path) -> tuple[Optional[Structure], Path]:
        """Read dump.lammpstrj."""
        dump_file = work_dir / "dump.lammpstrj"
        if not dump_file.exists():
            raise FileNotFoundError(f"Dump file not found at {dump_file}")

        # Read last frame
        atoms = ase.io.read(dump_file, index=-1, format="lammps-dump-text") # type: ignore[no-untyped-call]

        if isinstance(atoms, list):
             atoms = atoms[-1]

        structure = Structure.from_ase(atoms) # type: ignore[arg-type]

        return structure, dump_file
