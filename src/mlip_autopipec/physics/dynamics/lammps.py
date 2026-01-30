import logging
import subprocess
from pathlib import Path

import ase.io

from mlip_autopipec.domain_models.config import LammpsConfig, MDParams
from mlip_autopipec.domain_models.job import JobStatus, LammpsResult
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class LammpsRunner:
    """Wrapper for running LAMMPS simulations."""

    def __init__(self, config: LammpsConfig) -> None:
        self.config = config

    def run(self, structure: Structure, params: MDParams, work_dir: Path) -> LammpsResult:
        """
        Run a LAMMPS simulation.

        Args:
            structure: Initial structure.
            params: MD parameters.
            work_dir: Directory to run simulation in.

        Returns:
            LammpsResult object.
        """
        work_dir.mkdir(parents=True, exist_ok=True)
        job_id = work_dir.name

        try:
            # 1. Write Inputs
            self._write_inputs(work_dir, structure, params)

            # 2. Run LAMMPS
            command = self.config.command.split()
            command.extend(["-in", "in.lammps"])

            logger.info(f"Running LAMMPS in {work_dir}: {' '.join(command)}")

            # Use a log file
            log_file = work_dir / "log.lammps"

            # Run subprocess
            try:
                result = subprocess.run(
                    command,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )

                # Write stdout/stderr to log
                with log_file.open("w") as f:
                    f.write(result.stdout)
                    f.write("\nSTDERR:\n")
                    f.write(result.stderr)

                if result.returncode != 0:
                    return LammpsResult(
                        job_id=job_id,
                        status=JobStatus.FAILED,
                        work_dir=work_dir,
                        duration_seconds=0.0,
                        log_content=result.stderr[-500:],
                        final_structure=structure, # Return initial as fallback
                        trajectory_path=work_dir / "dump.lammpstrj"
                    )

            except subprocess.TimeoutExpired:
                logger.error(f"Job {job_id} timed out after {self.config.timeout}s")
                return LammpsResult(
                    job_id=job_id,
                    status=JobStatus.TIMEOUT,
                    work_dir=work_dir,
                    duration_seconds=float(self.config.timeout),
                    log_content="Timeout Expired",
                    final_structure=structure,
                    trajectory_path=work_dir / "dump.lammpstrj"
                )

            # 3. Parse Output
            final_structure = self._parse_output(work_dir / "dump.lammpstrj")

            # Rough duration (placeholder)
            duration = 0.0 # TODO: Calculate properly

            return LammpsResult(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                work_dir=work_dir,
                duration_seconds=duration,
                log_content=result.stdout[-500:],
                final_structure=final_structure,
                trajectory_path=work_dir / "dump.lammpstrj"
            )

        except Exception as e:
            logger.exception(f"Unexpected error in LammpsRunner: {e}")
            return LammpsResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                work_dir=work_dir,
                duration_seconds=0.0,
                log_content=str(e),
                final_structure=structure,
                trajectory_path=work_dir / "dump.lammpstrj"
            )

    def _write_inputs(self, work_dir: Path, structure: Structure, params: MDParams) -> None:
        """Write LAMMPS data and input script."""
        atoms = structure.to_ase()

        # Write data file
        ase.io.write(work_dir / "data.lammps", atoms, format="lammps-data") # type: ignore[no-untyped-call]

        # Generate input script
        # Using hybrid/overlay as requested by SPEC.
        # LJ/ZBL as baseline + ACE (placeholder since we don't have ACE files)

        script = f"""
units           metal
atom_style      atomic
boundary        p p p

read_data       data.lammps

# Hybrid pair style: LJ/Cut for baseline + ZBL + (ACE placeholder)
pair_style      hybrid/overlay lj/cut 2.5 zbl 2.0 2.5
pair_coeff      * * lj/cut 1.0 1.0 2.5
pair_coeff      * * zbl 14.0 14.0  # Si Z=14

mass            * 28.0855  # Si mass approx

velocity        all create {params.temperature} 12345 mom yes rot no

fix             1 all {params.ensemble} temp {params.temperature} {params.temperature} 0.1
"""
        if params.ensemble == "npt" and params.pressure is not None:
             script = script.replace(f"fix             1 all {params.ensemble}", f"fix             1 all npt temp {params.temperature} {params.temperature} 0.1 iso {params.pressure} {params.pressure} 1.0")

        script += f"""
timestep        {params.timestep}
thermo          100

dump            1 all custom 100 dump.lammpstrj id type x y z
run             {params.n_steps}
"""
        (work_dir / "in.lammps").write_text(script)

    def _parse_output(self, dump_path: Path) -> Structure:
        """Parse the final frame from LAMMPS dump using streaming read."""
        if not dump_path.exists():
            raise FileNotFoundError(f"Dump file not found: {dump_path}")

        # Use ASE iread to avoid loading full trajectory into memory
        last_atoms = None
        # type: ignore[no-untyped-call]
        for atoms in ase.io.iread(dump_path, format="lammps-dump-text"):
            last_atoms = atoms

        if last_atoms is None:
             raise ValueError("Dump file is empty")

        return Structure.from_ase(last_atoms) # type: ignore[arg-type]
