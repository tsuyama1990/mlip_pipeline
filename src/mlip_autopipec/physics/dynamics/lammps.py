import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import ase.io
from mlip_autopipec.domain_models.config import LammpsConfig, MDParams
from mlip_autopipec.domain_models.job import JobStatus, LammpsResult
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class LammpsRunner:
    def __init__(self, config: LammpsConfig):
        self.config = config

    def run(self, structure: Structure, params: MDParams, work_dir: Optional[Path] = None) -> LammpsResult:
        """
        Run a LAMMPS simulation.

        Args:
            structure: Initial structure.
            params: MD parameters.
            work_dir: Directory to run in. If None, uses a temporary directory.

        Returns:
            LammpsResult: The result of the simulation.
        """
        if work_dir:
             return self._execute_in_dir(work_dir, structure, params)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                return self._execute_in_dir(Path(tmp), structure, params)

    def _execute_in_dir(self, work_dir: Path, structure: Structure, params: MDParams) -> LammpsResult:
        work_dir.mkdir(parents=True, exist_ok=True)
        job_id = work_dir.name

        logger.info(f"Running LAMMPS in {work_dir}")

        try:
            self._write_inputs(work_dir, structure, params)
        except Exception as e:
            logger.exception("Failed to write LAMMPS inputs")
            return LammpsResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                work_dir=work_dir,
                duration_seconds=0.0,
                log_content=f"Input generation failed: {e}"
            )

        # Run LAMMPS
        stdout, stderr, returncode, duration = self._run_lammps(work_dir)

        log_content = stdout[-1000:] if stdout else "" # Store tail

        if returncode != 0:
            logger.error(f"LAMMPS failed with code {returncode}")
            return LammpsResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                work_dir=work_dir,
                duration_seconds=duration,
                log_content=stderr + "\n" + log_content
            )

        try:
            final_structure, trajectory_path = self._parse_output(work_dir)
            return LammpsResult(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                work_dir=work_dir,
                duration_seconds=duration,
                log_content=log_content,
                final_structure=final_structure,
                trajectory_path=trajectory_path
            )
        except Exception as e:
            logger.exception("Failed to parse LAMMPS output")
            return LammpsResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                work_dir=work_dir,
                duration_seconds=duration,
                log_content=f"Parsing failed: {e}\n{log_content}"
            )

    def _write_inputs(self, work_dir: Path, structure: Structure, params: MDParams) -> None:
        # Write data file
        atoms = structure.to_ase()
        ase.io.write(work_dir / "data.lammps", atoms, format="lammps-data") # type: ignore[no-untyped-call]

        # Write input script
        # Basic LJ potential for testing
        template = f"""
units           metal
atom_style      atomic
boundary        p p p

read_data       data.lammps

pair_style      lj/cut 2.5
pair_coeff      * * 1.0 1.0

velocity        all create {params.temperature} {self.config.seed if hasattr(self.config, 'seed') else 12345} dist gaussian

fix             1 all nvt temp {params.temperature} {params.temperature} 0.1

timestep        {params.timestep}
run             {params.n_steps}

dump            1 all custom 10 dump.lammpstrj id type x y z
dump_modify     1 sort id
"""
        (work_dir / "in.lammps").write_text(template)

    def _run_lammps(self, work_dir: Path) -> Tuple[str, str, int, float]:
        start = time.time()

        cmd = self.config.command.split()
        if self.config.cores > 1:
            cmd = ["mpirun", "-np", str(self.config.cores)] + cmd

        cmd.extend(["-in", "in.lammps"])

        try:
            res = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            return res.stdout, res.stderr, res.returncode, time.time() - start
        except subprocess.TimeoutExpired:
            return "", "Timeout", 124, time.time() - start
        except FileNotFoundError:
             return "", f"Executable not found: {cmd[0]}", 127, time.time() - start
        except Exception as e:
             return "", f"Execution error: {e}", 1, time.time() - start

    def _parse_output(self, work_dir: Path) -> Tuple[Structure, Path]:
        dump_file = work_dir / "dump.lammpstrj"
        if not dump_file.exists():
            raise FileNotFoundError(f"Dump file not found: {dump_file}")

        # Parse last frame
        atoms = ase.io.read(dump_file, index=-1, format="lammps-dump-text") # type: ignore[no-untyped-call]

        if isinstance(atoms, list):
             atoms = atoms[-1]

        if not isinstance(atoms, ase.Atoms):
             raise TypeError(f"Expected ase.Atoms, got {type(atoms)}")

        # If chemical symbols are missing (defaults to X), we might want to fix them.
        # But for generic runner, we might not know mapping.
        # Structure model requires symbols.

        return Structure.from_ase(atoms), dump_file
