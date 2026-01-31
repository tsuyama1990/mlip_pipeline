import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import ase.io
import ase.data
from mlip_autopipec.domain_models import (
    JobStatus,
    LammpsConfig,
    LammpsResult,
    MDParams,
    Structure,
    PotentialConfig,
)
from mlip_autopipec.physics.dynamics.log_parser import LammpsLogParser


class LammpsRunner:
    """
    Executes LAMMPS MD simulations.
    """

    def __init__(
        self,
        config: LammpsConfig,
        potential_config: PotentialConfig,
        base_work_dir: Path = Path("_work_md"),
    ):
        self.config = config
        self.potential_config = potential_config
        self.base_work_dir = base_work_dir
        self.base_work_dir.mkdir(parents=True, exist_ok=True)
        self.log_parser = LammpsLogParser()

    def run(
        self,
        structure: Structure,
        params: MDParams,
        potential_path: Optional[Path] = None,
    ) -> LammpsResult:
        """
        Run a single MD simulation.
        """
        job_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = self.base_work_dir / f"job_{timestamp}_{job_id[:8]}"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Write Inputs
            self._write_inputs(work_dir, structure, params, potential_path)

            # 2. Execute
            start_time = datetime.now()
            log_content = self._execute(work_dir)
            duration = (datetime.now() - start_time).total_seconds()

            # 3. Parse Output
            final_structure, trajectory_path, max_gamma = self._parse_output(
                work_dir, structure
            )

            return LammpsResult(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                work_dir=work_dir,
                duration_seconds=duration,
                log_content=log_content,
                final_structure=final_structure,
                trajectory_path=trajectory_path,
                max_gamma=max_gamma,
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
                log_content = f"Command failed.\nStderr: {e.stderr}"

            # Try to parse output even if failed (e.g. fix halt with error)
            try:
                final_structure, trajectory_path, max_gamma = self._parse_output(
                    work_dir, structure
                )
            except Exception:
                final_structure = structure
                trajectory_path = work_dir / "dump.lammpstrj"
                max_gamma = None

            # Check if it was a halt
            status = JobStatus.FAILED
            # If parsing detected halt or max_gamma is high, we might consider this expected.

            return LammpsResult(
                job_id=job_id,
                status=status,
                work_dir=work_dir,
                duration_seconds=0.0,
                log_content=log_content,
                final_structure=final_structure,
                trajectory_path=trajectory_path,
                max_gamma=max_gamma,
            )
        except Exception as e:
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
        self,
        work_dir: Path,
        structure: Structure,
        params: MDParams,
        potential_path: Optional[Path],
    ) -> None:
        """Write data.lammps and in.lammps."""
        # Write Structure
        atoms = structure.to_ase()
        ase.io.write(work_dir / "data.lammps", atoms, format="lammps-data")  # type: ignore[no-untyped-call]

        unique_elements = sorted(list(set(structure.symbols)))

        # Interaction
        pair_style = ""
        pair_coeff = ""

        if potential_path and self.potential_config.pair_style == "hybrid/overlay":
             zbl_in = self.potential_config.zbl_inner_cutoff
             zbl_out = self.potential_config.zbl_outer_cutoff

             pair_style = f"pair_style      hybrid/overlay pace zbl {zbl_in} {zbl_out}"
             pot_file_str = str(potential_path.resolve())

             elem_str = " ".join(unique_elements)
             pair_coeff += f"pair_coeff      * * pace {pot_file_str} {elem_str}\n"

             for i, el1 in enumerate(unique_elements):
                 z1 = ase.data.atomic_numbers[el1]
                 for j, el2 in enumerate(unique_elements):
                     if j < i:
                         continue
                     z2 = ase.data.atomic_numbers[el2]
                     pair_coeff += f"pair_coeff      {i+1} {j+1} zbl {z1} {z2}\n"

        elif potential_path:
             pair_style = "pair_style      pace"
             pot_file_str = str(potential_path.resolve())
             elem_str = " ".join(unique_elements)
             pair_coeff = f"pair_coeff      * * pace {pot_file_str} {elem_str}"
        else:
             pair_style = "pair_style      lj/cut 2.5"
             pair_coeff = "pair_coeff      * * 1.0 1.0"

        # UQ / Watchdog
        uq_cmds = ""
        dump_vars = "id type x y z"
        thermo_custom = "step temp pe"

        if params.uncertainty_threshold is not None and potential_path:
            pot_file_str = str(potential_path.resolve())
            uq_cmds = f"""
# Uncertainty Quantification
compute         pace_gamma all pace {pot_file_str}
variable        max_gamma equal max(c_pace_gamma)
fix             watchdog all halt 10 v_max_gamma > {params.uncertainty_threshold} error hard
"""
            dump_vars += " c_pace_gamma"
            thermo_custom += " v_max_gamma"

        # Ensemble
        fix_cmd = ""
        if params.ensemble == "NVT":
            fix_cmd = f"fix 1 all nvt temp {params.temperature} {params.temperature} $(100.0*dt)"
        elif params.ensemble == "NPT":
            pres = params.pressure if params.pressure is not None else 0.0
            fix_cmd = f"fix 1 all npt temp {params.temperature} {params.temperature} $(100.0*dt) iso {pres} {pres} $(1000.0*dt)"

        input_script = f"""
# Cycle 02/03 - MD
units           metal
atom_style      atomic
boundary        p p p

read_data       data.lammps

# Interaction
{pair_style}
{pair_coeff}

timestep        {params.timestep}

# Output
dump            1 all custom 100 dump.lammpstrj {dump_vars}
thermo          10
thermo_style    custom {thermo_custom}
log             log.lammps

# UQ
{uq_cmds}

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

        cmd_list = cmd_str.split() + ["-in", "in.lammps"]

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
    ) -> tuple[Structure, Path, Optional[float]]:
        """Parse trajectory and return final structure, path, and max_gamma."""
        traj_path = work_dir / "dump.lammpstrj"
        log_path = work_dir / "log.lammps"

        max_gamma = None

        # Parse log for max_gamma and halt status
        if log_path.exists():
            log_content = log_path.read_text()
            parse_result = self.log_parser.parse(log_content)
            max_gamma = parse_result.max_gamma

            # If halt detected but max_gamma not parsed (maybe thermo didn't print in time?),
            # assume it triggered threshold.
            if parse_result.halt_detected and max_gamma is None:
                # We assume the last step triggered it.
                # Just return a flag value or None (Orchestrator can handle halt logic separately via JobStatus?)
                # But Orchestrator logic: if result.max_gamma > threshold -> Detect.
                # So we must return > threshold.
                max_gamma = 999.9

        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory {traj_path} not found.")

        # Read last frame
        atoms = ase.io.read(traj_path, index=-1, format="lammps-dump-text")  # type: ignore[no-untyped-call]

        if isinstance(atoms, list):
            atoms = atoms[-1]

        # Restore symbols
        if len(atoms) == len(original_structure.symbols):
            atoms.set_chemical_symbols(original_structure.symbols)  # type: ignore[no-untyped-call]

        return Structure.from_ase(atoms), traj_path, max_gamma
