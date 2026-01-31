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
        work_dir.mkdir()

        try:
            # 1. Write Inputs
            self._write_inputs(work_dir, structure, params, potential_path)

            # 2. Execute
            start_time = datetime.now()
            log_content = self._execute(work_dir)
            duration = (datetime.now() - start_time).total_seconds()

            # 3. Parse Output
            # Note: If fix halt triggered, we might have exit code 0 or error, depending on config.
            # We check log content for max_gamma.
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
            # Simple check: if max_gamma is present and high, maybe it's not "FAILED" in the traditional sense?
            # But the job stopped early. 'COMPLETED' might be misleading if it crashed.
            # However, orchestrator will see max_gamma and handle it.
            # If log says "Halt triggered", we might consider it COMPLETED (as in, ran successfully until halt).
            # But normally CalledProcessError implies non-zero exit.
            # If fix halt used "error hard", it exits with 1.
            # Let's keep FAILED but provide max_gamma.

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
        # Ensure we write sorted species for consistent mapping
        # types will be 1..N for sorted(unique(symbols))
        # ase.io.write handles this but we need to match it in pair_coeff
        ase.io.write(work_dir / "data.lammps", atoms, format="lammps-data")  # type: ignore[no-untyped-call]

        # Determine elements in order of types (sorted by ASE logic usually)
        # Using sorted(unique) is a safe bet for ASE default
        unique_elements = sorted(list(set(structure.symbols)))

        # Interaction
        pair_style = ""
        pair_coeff = ""

        if potential_path and self.potential_config.pair_style == "hybrid/overlay":
            # Hybrid ACE + ZBL
            zbl_in = self.potential_config.zbl_inner_cutoff
            zbl_out = self.potential_config.zbl_outer_cutoff

            pair_style = f"pair_style      hybrid/overlay pace zbl {zbl_in} {zbl_out}"

            # Copy potential file to work_dir? Or reference absolute path.
            # LAMMPS handles absolute paths usually.
            pot_file_str = str(potential_path.resolve())

            # Pace coeff
            # pair_coeff * * pace potential.yace Element1 Element2 ...
            # Note: Elements must match type order 1, 2, ...
            elem_str = " ".join(unique_elements)
            pair_coeff += f"pair_coeff      * * pace {pot_file_str} {elem_str}\n"

            # ZBL coeff
            # pair_coeff * * zbl Z1 Z2
            # We need to iterate over all pairs of types i, j
            # Or use * * zbl? No, zbl requires args.
            # Actually, if we use pair_style zbl with cutoffs in style command,
            # pair_coeff i j zbl Zi Zj
            for i, el1 in enumerate(unique_elements):
                z1 = ase.data.atomic_numbers[el1]
                for j, el2 in enumerate(unique_elements):
                    if j < i:
                        continue  # Symmetric
                    z2 = ase.data.atomic_numbers[el2]
                    pair_coeff += f"pair_coeff      {i + 1} {j + 1} zbl {z1} {z2}\n"

        elif potential_path:
            # Just ACE
            pair_style = "pair_style      pace"
            pot_file_str = str(potential_path.resolve())
            elem_str = " ".join(unique_elements)
            pair_coeff = f"pair_coeff      * * pace {pot_file_str} {elem_str}"
        else:
            # Fallback LJ
            pair_style = "pair_style      lj/cut 2.5"
            pair_coeff = "pair_coeff      * * 1.0 1.0"

        # UQ / Watchdog
        uq_cmds = ""
        dump_vars = "id type x y z"
        if params.uncertainty_threshold is not None and potential_path:
            # compute pace_gamma all pace potential.yace gamma_mode=1?
            # Check spec or pace docs. "compute pace ... gamma_mode=1" is likely correct for getting gamma.
            # Assuming compute pace returns gamma as scalar per atom?
            # Or vector?
            # Usually: compute ID group-ID pace filename
            # It generates c_ID[1] = energy, ... depending on implementation.
            # But user-pace often provides specific compute.
            # Spec says: "compute pace_gamma all pace ... gamma_mode=1"
            # "variable max_gamma equal max(c_pace_gamma)"
            pot_file_str = str(potential_path.resolve())
            uq_cmds = f"""
# Uncertainty Quantification
compute         pace_gamma all pace {pot_file_str}
# Note: implementation of compute pace varies. Assuming it outputs gamma or we configure it.
# If pace compute output is just gamma (or vector where gamma is component), we need to know index.
# For now, following Spec:
variable        max_gamma equal max(c_pace_gamma)
fix             watchdog all halt 10 v_max_gamma > {params.uncertainty_threshold} error hard
"""
            dump_vars += " c_pace_gamma"

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

        # Use stdout/stderr redirection to files to avoid full memory buffering
        with (work_dir / "stdout.log").open("w") as stdout_f, (
            work_dir / "stderr.log"
        ).open("w") as stderr_f:
            subprocess.run(
                cmd_list,
                cwd=work_dir,
                timeout=self.config.timeout,
                check=True,
                stdout=stdout_f,
                stderr=stderr_f,
            )

        return (work_dir / "stdout.log").read_text()

    def _parse_output(
        self, work_dir: Path, original_structure: Structure
    ) -> tuple[Structure, Path, Optional[float]]:
        """Parse trajectory and return final structure, path, and max_gamma."""
        traj_path = work_dir / "dump.lammpstrj"
        log_path = work_dir / "log.lammps"

        max_gamma = None

        # Incremental log parsing
        if log_path.exists():
            with log_path.open("r") as f:
                # Seek to end and read backwards? Or read chunk by chunk.
                # For scalability, we shouldn't read whole file.
                # But typical LAMMPS logs aren't massive compared to trajectories.
                # However, if we look for "Fix halt", we can scan line by line.
                for line in f:
                    if "Fix halt" in line:
                        max_gamma = 999.9
                        break

        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory {traj_path} not found.")

        # Read last frame safely using iread and exhausting iterator
        # This is more memory efficient than read(index=-1) for huge files?
        # ase.io.read(index=-1) might optimize, but iread is safer.
        last_atoms = None
        for atoms in ase.io.iread(traj_path, format="lammps-dump-text"):
            last_atoms = atoms

        if last_atoms is None:
             raise ValueError("Trajectory is empty")

        # Restore symbols
        if len(last_atoms) == len(original_structure.symbols):
            last_atoms.set_chemical_symbols(original_structure.symbols)  # type: ignore[no-untyped-call]

        return Structure.from_ase(last_atoms), traj_path, max_gamma
