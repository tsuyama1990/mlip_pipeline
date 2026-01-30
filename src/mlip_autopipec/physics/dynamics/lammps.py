import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

import ase.io
from mlip_autopipec.domain_models.config import LammpsConfig, MDParams
from mlip_autopipec.domain_models.job import JobStatus, LammpsResult
from mlip_autopipec.domain_models.structure import Structure


class LammpsRunner:
    """
    Executes LAMMPS MD simulations.
    """

    def __init__(self, config: LammpsConfig, base_work_dir: Path = Path("_work_md")):
        self.config = config
        self.base_work_dir = base_work_dir
        self.base_work_dir.mkdir(parents=True, exist_ok=True)

    def run(self, structure: Structure, params: MDParams) -> LammpsResult:
        """
        Run a single MD simulation.
        """
        job_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = self.base_work_dir / f"job_{timestamp}_{job_id[:8]}"
        work_dir.mkdir()

        try:
            # 1. Write Inputs
            self._write_inputs(work_dir, structure, params)

            # 2. Execute
            start_time = datetime.now()
            log_content = self._execute(work_dir)
            duration = (datetime.now() - start_time).total_seconds()

            # 3. Parse Output
            final_structure, trajectory_path = self._parse_output(work_dir, structure)

            return LammpsResult(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                work_dir=work_dir,
                duration_seconds=duration,
                log_content=log_content,
                final_structure=final_structure,
                trajectory_path=trajectory_path,
            )

        except subprocess.TimeoutExpired:
            return LammpsResult(
                job_id=job_id,
                status=JobStatus.TIMEOUT,
                work_dir=work_dir,
                duration_seconds=self.config.timeout,
                log_content="Timeout Expired",
                final_structure=structure,  # Return initial as fallback
                trajectory_path=work_dir / "dump.lammpstrj",
                max_gamma=None,
            )
        except subprocess.CalledProcessError as e:
            log_file = work_dir / "log.lammps"
            if log_file.exists():
                log_content = log_file.read_text()
            else:
                # Capture stderr if log doesn't exist
                log_content = f"Command failed.\nStderr: {e.stderr}"

            return LammpsResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                work_dir=work_dir,
                duration_seconds=0.0,
                log_content=log_content,
                final_structure=structure,
                trajectory_path=work_dir / "dump.lammpstrj",
                max_gamma=None,
            )
        except Exception as e:
            # Try to read log if it exists
            log_file = work_dir / "log.lammps"
            log_content = log_file.read_text() if log_file.exists() else str(e)

            return LammpsResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                work_dir=work_dir,
                duration_seconds=0.0,
                log_content=log_content,
                final_structure=structure,
                trajectory_path=work_dir / "dump.lammpstrj",
                max_gamma=None,
            )

    def _write_inputs(
        self, work_dir: Path, structure: Structure, params: MDParams
    ) -> None:
        """Write data.lammps and in.lammps."""
        # Write Structure
        atoms = structure.to_ase()
        ase.io.write(work_dir / "data.lammps", atoms, format="lammps-data")  # type: ignore[no-untyped-call]

        # Template for LJ (for Cycle 02 testing)
        # In real scenario, we would use a potential file. Here we use internal LJ.

        # Determine barostat/thermostat commands
        fix_cmd = ""
        if params.ensemble == "NVT":
            fix_cmd = f"fix 1 all nvt temp {params.temperature} {params.temperature} $(100.0*dt)"
        elif params.ensemble == "NPT":
            pres = params.pressure if params.pressure is not None else 0.0
            fix_cmd = f"fix 1 all npt temp {params.temperature} {params.temperature} $(100.0*dt) iso {pres} {pres} $(1000.0*dt)"

        input_script = f"""
# Cycle 02 - One Shot MD
units           metal
atom_style      atomic
boundary        p p p

read_data       data.lammps

# Interaction (Lennard-Jones for testing)
pair_style      lj/cut 2.5
pair_coeff      * * 1.0 1.0

timestep        {params.timestep}

# Output
dump            1 all custom 100 dump.lammpstrj id type x y z
log             log.lammps

# Ensemble
{fix_cmd}

run             {params.n_steps}
"""
        (work_dir / "in.lammps").write_text(input_script)

    def _execute(self, work_dir: Path) -> str:
        """Execute LAMMPS subprocess."""
        cmd_str = self.config.command
        if self.config.use_mpi:
            cmd_str = f"{self.config.mpi_command} {cmd_str}"

        # Add input argument
        cmd_list = cmd_str.split() + ["-in", "in.lammps"]

        # Check executable existence if not using mpirun (mpirun wraps it)
        # But simplistic check:
        exe = cmd_list[0]
        if not shutil.which(exe):
            raise FileNotFoundError(f"Executable '{exe}' not found.")

        result = subprocess.run(
            cmd_list,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=self.config.timeout,
            check=True,
        )

        return result.stdout

    def _parse_output(
        self, work_dir: Path, original_structure: Structure
    ) -> tuple[Structure, Path]:
        """Parse trajectory and return final structure."""
        traj_path = work_dir / "dump.lammpstrj"
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory {traj_path} not found.")

        # Scalability optimization: Use seeking to read only the last frame.
        # LAMMPS dump format usually ends with:
        # ITEM: TIMESTEP
        # ...
        # ITEM: NUMBER OF ATOMS
        # ...
        # ITEM: BOX BOUNDS ...
        # ...
        # ITEM: ATOMS ...

        # We need to find the last occurrence of "ITEM: TIMESTEP".
        # We can read chunks from the end.

        # Helper to get last frame content
        last_frame_content = self._read_last_frame_optimized(traj_path)

        # Parse content using string IO
        from io import StringIO
        # Use StringIO to feed ASE
        with StringIO(last_frame_content) as f:
             # Using 'lammps-dump-text' on string buffer
             # ASE expects file-like object
             atoms = ase.io.read(f, format="lammps-dump-text") # type: ignore[no-untyped-call]

        if isinstance(atoms, list):
            atoms = atoms[-1]

        if atoms is None:
             raise ValueError("Failed to parse last frame from trajectory")

        # Restore symbols from original structure
        # (Assuming atom order hasn't changed, which is true for standard MD)
        if len(atoms) == len(original_structure.symbols):
            atoms.set_chemical_symbols(original_structure.symbols)  # type: ignore[no-untyped-call]

        return Structure.from_ase(atoms), traj_path

    def _read_last_frame_optimized(self, filepath: Path, chunk_size: int = 1024 * 1024) -> str:
        """
        Reads the last frame from a LAMMPS dump file using backward seeking.
        """
        with open(filepath, "rb") as f:
            f.seek(0, 2) # Seek to end
            file_size = f.tell()

            buffer = b""
            # Loop backwards
            for pos in range(file_size, -1, -chunk_size):
                start = max(0, pos - chunk_size)
                f.seek(start)
                chunk = f.read(pos - start)
                buffer = chunk + buffer

                # Check for at least two "ITEM: TIMESTEP" if possible to identify boundaries,
                # or just one if it's near the end.
                # Actually, we need the last "ITEM: TIMESTEP".
                # But "ITEM: TIMESTEP" might appear in the middle of the chunk.
                # We want the LAST occurrence in the file.

                # Search for marker
                marker = b"ITEM: TIMESTEP"
                count = buffer.count(marker)

                if count >= 1:
                     # Find index of the last marker
                     idx = buffer.rfind(marker)

                     # Check if we have the full frame.
                     # A frame ends at the end of the file or next ITEM: TIMESTEP
                     # Since we are reading from end, if we found the marker and we have the end of file, we likely have the frame.
                     # However, to be safe, we might need to ensure we read enough bytes after the marker.
                     # But buffer includes everything from marker to EOF (plus some prefix).

                     # If count > 1, we definitely have the full last frame (between last marker and EOF).
                     # If count == 1, and we haven't reached start of file, we *might* have the full frame if the chunk was big enough.
                     # But if the frame is HUGE (bigger than chunk), we might need to keep reading back until we find the *previous* marker
                     # to know where this one starts? No, we just need from "ITEM: TIMESTEP" to EOF.

                     # We assume the last "ITEM: TIMESTEP" in the file starts the last frame.
                     # So if we found it, we just take everything from there.

                     return buffer[idx:].decode("utf-8", errors="replace")

                if start == 0:
                    break

            # If we reached start and buffer has content but no marker (maybe file is small/malformed or single frame without marker?)
            # Just return whole buffer
            return buffer.decode("utf-8", errors="replace")
