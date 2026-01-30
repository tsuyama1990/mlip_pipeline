import subprocess
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict
import ase
import ase.io

from mlip_autopipec.domain_models.config import LammpsConfig
from mlip_autopipec.domain_models.job import JobStatus, LammpsResult
from mlip_autopipec.domain_models.structure import Structure


class MDParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float = 300.0
    pressure: float = 0.0  # Bar?
    n_steps: int = 1000
    timestep: float = 0.001 # ps
    ensemble: str = "nvt" # nvt, npt


class LammpsRunner:
    def __init__(self, config: LammpsConfig):
        self.config = config

    def run(self, structure: Structure, work_dir: Path, params: MDParams) -> LammpsResult:
        """
        Run LAMMPS simulation.
        """
        # Ensure work_dir exists
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. Write inputs
        self._write_inputs(work_dir, structure, params)

        # 2. Execute
        cmd = f"{self.config.command} -in in.lammps"

        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )

            # Check return code
            if proc.returncode != 0:
                return LammpsResult(
                    job_id=work_dir.name,
                    status=JobStatus.FAILED,
                    work_dir=work_dir,
                    duration_seconds=0.0,
                    log_content=proc.stderr[-1000:] + "\n" + proc.stdout[-1000:]
                )

            # 3. Parse output
            return self._parse_output(work_dir, proc.stdout, structure)

        except subprocess.TimeoutExpired:
            return LammpsResult(
                job_id=work_dir.name,
                status=JobStatus.TIMEOUT,
                work_dir=work_dir,
                duration_seconds=float(self.config.timeout),
                log_content="Timeout expired"
            )
        except Exception as e:
            # Catch all logic errors to prevent crash
            return LammpsResult(
                job_id=work_dir.name,
                status=JobStatus.FAILED,
                work_dir=work_dir,
                duration_seconds=0.0,
                log_content=str(e)
            )

    def _write_inputs(self, work_dir: Path, structure: Structure, params: MDParams) -> None:
        """Write data.lammps and in.lammps."""
        # Write data file
        atoms = structure.to_ase()
        ase.io.write(work_dir / "data.lammps", atoms, format="lammps-data") # type: ignore[no-untyped-call]

        # Write input script
        # Using a simple template for One-Shot (LJ potential)
        template = f"""
units           metal
atom_style      atomic
boundary        p p p

read_data       data.lammps

pair_style      lj/cut 2.5
pair_coeff      * * 1.0 1.0

thermo          100
thermo_style    custom step temp pe ke etotal press vol

timestep        {params.timestep}

# Dump
dump            1 all custom 100 dump.lammpstrj id type x y z
dump_modify     1 sort id

# Integration
fix             1 all {params.ensemble} temp {params.temperature} {params.temperature} $(100.0*dt)
"""

        if params.ensemble == "npt":
             template += f" iso {params.pressure} {params.pressure} $(1000.0*dt)\n"

        template += f"""
run             {params.n_steps}
"""

        (work_dir / "in.lammps").write_text(template)

    def _parse_output(self, work_dir: Path, stdout: str, initial_structure: Structure) -> LammpsResult:
        """Parse dump file."""
        dump_file = work_dir / "dump.lammpstrj"

        if not dump_file.exists():
             return LammpsResult(
                job_id=work_dir.name,
                status=JobStatus.FAILED,
                work_dir=work_dir,
                duration_seconds=0.0,
                log_content="dump.lammpstrj not found.\n" + stdout[-1000:]
            )

        try:
            # Read last frame
            traj = ase.io.read(dump_file, index=-1, format="lammps-dump-text") # type: ignore[no-untyped-call]

            if isinstance(traj, list):
                atoms = traj[-1]
            else:
                atoms = traj

            atoms = cast(ase.Atoms, atoms)

            # Restore symbols
            atoms.set_pbc(initial_structure.pbc) # type: ignore[no-untyped-call]

            # Assuming types match 1-to-1 with initial symbols set if sorted by ID.
            if len(atoms) == len(initial_structure.symbols):
                 atoms.set_chemical_symbols(initial_structure.symbols) # type: ignore[no-untyped-call]

            final_structure = Structure.from_ase(atoms)

            return LammpsResult(
                job_id=work_dir.name,
                status=JobStatus.COMPLETED,
                work_dir=work_dir,
                duration_seconds=0.0,
                log_content=stdout[-1000:],
                final_structure=final_structure,
                trajectory_path=dump_file
            )
        except Exception as e:
             return LammpsResult(
                job_id=work_dir.name,
                status=JobStatus.FAILED,
                work_dir=work_dir,
                duration_seconds=0.0,
                log_content=f"Parsing failed: {e}\n" + stdout[-1000:]
            )
