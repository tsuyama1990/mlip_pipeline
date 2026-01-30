import logging
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import ase.io
import ase
from ase.data import atomic_numbers
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

        # Determine species mapping (ASE lammps-data sorts species alphabetically)
        species = sorted(list(set(atoms.get_chemical_symbols()))) # type: ignore[no-untyped-call]

        # Generate pair coefficients for ZBL
        zbl_coeffs = []
        for i, s1 in enumerate(species, 1):
            for j, s2 in enumerate(species, 1):
                if i <= j: # Symmetric
                    z1 = atomic_numbers[s1]
                    z2 = atomic_numbers[s2]
                    zbl_coeffs.append(f"pair_coeff      {i} {j} zbl {z1} {z2}")

        pair_cmds = "\n".join(zbl_coeffs)

        # Write input script
        # Hybrid potential: LJ/ZBL baseline
        template = f"""
units           metal
atom_style      atomic
boundary        p p p

read_data       data.lammps

pair_style      hybrid/overlay lj/cut 2.5 zbl 4.0 5.0
pair_coeff      * * lj/cut 1.0 1.0
{pair_cmds}

velocity        all create {params.temperature} {self.config.seed} dist gaussian

fix             1 all nvt temp {params.temperature} {params.temperature} 0.1

timestep        {params.timestep}
run             {params.n_steps}

dump            1 all custom 10 dump.lammpstrj id type x y z
dump_modify     1 sort id
"""
        (work_dir / "in.lammps").write_text(template)

    def _run_lammps(self, work_dir: Path) -> Tuple[str, str, int, float]:
        start = time.time()

        # Security: Use shlex to split command correctly
        cmd = shlex.split(self.config.command)

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

        # Optimize parsing: Read file backwards to find the last TIMESTEP
        # This implementation avoids full file scan.

        try:
            last_frame_content = self._read_last_frame_text(dump_file)

            # Use temp file to parse with ASE to reuse parser logic
            with tempfile.NamedTemporaryFile(mode='w+', suffix=".lammpstrj") as tmp:
                tmp.write(last_frame_content)
                tmp.flush()
                tmp.seek(0)
                atoms = ase.io.read(tmp.name, format="lammps-dump-text") # type: ignore[no-untyped-call]

            if not isinstance(atoms, ase.Atoms):
                 raise TypeError(f"Expected ase.Atoms, got {type(atoms)}")

            return Structure.from_ase(atoms), dump_file

        except Exception as e:
            # Fallback to iread if custom parser fails
            logger.warning(f"Optimized parser failed ({e}), falling back to standard iterator")
            last_atoms = None
            for atoms in ase.io.iread(dump_file, format="lammps-dump-text"): # type: ignore[no-untyped-call]
                last_atoms = atoms

            if last_atoms is None:
                 raise ValueError("No frames found in dump file")

            return Structure.from_ase(last_atoms), dump_file

    def _read_last_frame_text(self, filepath: Path, chunk_size: int = 4096) -> str:
        """
        Read the last frame from a LAMMPS dump file by seeking backwards.
        """
        with filepath.open('rb') as f:
            f.seek(0, 2)
            file_size = f.tell()

            buffer = b""
            pointer = file_size
            frames_found = 0

            while pointer > 0:
                read_size = min(chunk_size, pointer)
                pointer -= read_size
                f.seek(pointer)
                chunk = f.read(read_size)
                buffer = chunk + buffer

                # Check for "ITEM: TIMESTEP"
                # If we find 2 occurrences, we have the start of the last frame and the start of the previous
                # We want from the last "ITEM: TIMESTEP" to end.

                markers = buffer.count(b"ITEM: TIMESTEP")
                if markers >= 1:
                     # Found at least one start marker.
                     # We need to ensure we have the WHOLE last frame.
                     # If we found 2, we definitely have the last frame fully in buffer (assuming it's after the second-to-last marker).
                     # Actually, we just need to find the *last* occurrence of "ITEM: TIMESTEP" in the file.

                     last_marker_idx = buffer.rfind(b"ITEM: TIMESTEP")
                     if last_marker_idx != -1:
                         return buffer[last_marker_idx:].decode('utf-8')

        raise ValueError("Could not find start of frame in file")
