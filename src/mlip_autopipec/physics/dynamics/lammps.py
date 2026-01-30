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
                trajectory_path=trajectory_path
            )

        except subprocess.TimeoutExpired:
            return LammpsResult(
                job_id=job_id,
                status=JobStatus.TIMEOUT,
                work_dir=work_dir,
                duration_seconds=self.config.timeout,
                log_content="Timeout Expired",
                final_structure=structure, # Return initial as fallback
                trajectory_path=work_dir / "dump.lammpstrj",
                max_gamma=None
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
                max_gamma=None
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
                max_gamma=None
            )

    def _write_inputs(self, work_dir: Path, structure: Structure, params: MDParams) -> None:
        """Write data.lammps and in.lammps."""
        # Write Structure
        atoms = structure.to_ase()
        ase.io.write(work_dir / "data.lammps", atoms, format="lammps-data") # type: ignore[no-untyped-call]

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
            check=True
        )

        return result.stdout

    def _parse_output(self, work_dir: Path, original_structure: Structure) -> tuple[Structure, Path]:
        """Parse trajectory and return final structure."""
        traj_path = work_dir / "dump.lammpstrj"
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory {traj_path} not found.")

        # Read last frame
        # ase.io.read returns Atoms or list of Atoms. index=-1 returns single Atoms.
        atoms = ase.io.read(traj_path, index=-1, format="lammps-dump-text") # type: ignore[no-untyped-call]

        if isinstance(atoms, list):
             # Should not happen with index=-1 but type guard
             atoms = atoms[-1]

        # Restore symbols from original structure
        # (Assuming atom order hasn't changed, which is true for standard MD)
        if len(atoms) == len(original_structure.symbols):
            atoms.set_chemical_symbols(original_structure.symbols) # type: ignore[no-untyped-call]

        return Structure.from_ase(atoms), traj_path
