import shutil
import subprocess
import uuid
import shlex
from datetime import datetime
from pathlib import Path
from typing import Optional

import ase.io
import ase.data
from mlip_autopipec.domain_models.dynamics import (
    LammpsConfig,
    LammpsResult,
    MDParams,
)
from mlip_autopipec.domain_models.config import PotentialConfig
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.structure import Structure
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

        # Validate command immediately upon instantiation
        # This ensures we don't start setting up a job if the command is invalid
        LammpsConfig.validate_command(self.config.command)
        if self.config.use_mpi:
            LammpsConfig.validate_command(self.config.mpi_command)

    def run(
        self,
        structure: Structure,
        params: MDParams,
        potential_path: Optional[Path] = None,
        extra_commands: Optional[list[str]] = None,
    ) -> LammpsResult:
        """
        Run a single MD simulation.
        """
        self._validate_params(params)

        job_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = self.base_work_dir / f"job_{timestamp}_{job_id[:8]}"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Write Inputs
            self._write_inputs(work_dir, structure, params, potential_path, extra_commands)

            # 2. Execute
            start_time = datetime.now()
            self._execute(work_dir)
            duration = (datetime.now() - start_time).total_seconds()

            # 3. Parse Output
            final_structure, trajectory_path, max_gamma = self._parse_output(
                work_dir, structure
            )

            # Read tail of log content for debugging
            log_file = work_dir / "stdout.log"
            log_content = self._read_log_tail(log_file)

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
            log_content = self._collect_log_content(work_dir)

            # Fallback to exception info if logs are empty
            if not log_content.strip():
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
            log_content = self._collect_log_content(work_dir)

            if not log_content.strip():
                log_content = str(e)

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
        extra_commands: Optional[list[str]] = None,
    ) -> None:
        """Write data.lammps and in.lammps."""
        # Write Structure
        atoms = structure.to_ase()
        ase.io.write(work_dir / "data.lammps", atoms, format="lammps-data")  # type: ignore[no-untyped-call]

        unique_elements = sorted(list(set(structure.symbols)))

        # Interaction
        pair_style, pair_coeff = self._generate_potential_commands(
            unique_elements, potential_path
        )

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

        # Extra Commands
        extras = "\n".join(extra_commands) if extra_commands else ""

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

# Extra Commands (e.g. fix atom/swap)
{extras}

run             {params.n_steps}
"""
        self._validate_input_script(input_script)
        (work_dir / "in.lammps").write_text(input_script)

    def _validate_params(self, params: MDParams) -> None:
        """Validate MD parameters beyond Pydantic checks."""
        if params.timestep <= 0:
            raise ValueError("Timestep must be positive.")
        if params.temperature < 0:
            raise ValueError("Temperature cannot be negative (unless exotic physics intended, assuming error).")

    def _validate_input_script(self, script: str) -> None:
        """Basic validation of generated LAMMPS script."""
        if "pair_style" not in script:
            raise ValueError("Generated script missing 'pair_style'.")
        if "run" not in script:
            raise ValueError("Generated script missing 'run' command.")
        # Check for potential injection (simple check)
        forbidden = ["shell", "external", "python"]
        for line in script.splitlines():
            words = line.split()
            if words and words[0] in forbidden:
                raise ValueError(f"Forbidden command '{words[0]}' found in generated script.")

    def _generate_potential_commands(
        self, unique_elements: list[str], potential_path: Optional[Path]
    ) -> tuple[str, str]:
        """Generate pair_style and pair_coeff commands."""
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

        return pair_style, pair_coeff

    def _execute(self, work_dir: Path) -> None:
        """Execute LAMMPS subprocess, streaming output to file."""
        cmd_str = self.config.command
        if self.config.use_mpi:
            cmd_str = f"{self.config.mpi_command} {cmd_str}"

        # Safe splitting
        cmd_list = shlex.split(cmd_str) + ["-in", "in.lammps"]

        exe = cmd_list[0]
        if not shutil.which(exe):
            raise FileNotFoundError(f"Executable '{exe}' not found.")

        # Stream stdout/stderr to files
        stdout_path = work_dir / "stdout.log"
        stderr_path = work_dir / "stderr.log"

        # Explicitly stream output in chunks to avoid memory buffer issues
        with open(stdout_path, "wb") as f_out, open(stderr_path, "wb") as f_err:
            process = subprocess.Popen(
                cmd_list,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            try:
                # Manual streaming loop
                import select

                # Check for output until process ends
                while True:
                    reads = [process.stdout.fileno(), process.stderr.fileno()]
                    ret = select.select(reads, [], [], 0.5) # Timeout 0.5s to check process poll

                    for fd in ret[0]:
                        if fd == process.stdout.fileno():
                            chunk = process.stdout.read(4096)
                            if chunk:
                                f_out.write(chunk)
                                f_out.flush()
                        elif fd == process.stderr.fileno():
                            chunk = process.stderr.read(4096)
                            if chunk:
                                f_err.write(chunk)
                                f_err.flush()

                    if process.poll() is not None:
                         # Process ended, drain remaining
                         remaining_out = process.stdout.read()
                         if remaining_out:
                             f_out.write(remaining_out)
                         remaining_err = process.stderr.read()
                         if remaining_err:
                             f_err.write(remaining_err)
                         break

                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd_list)

            except Exception:
                # Cleanup if error
                process.kill()
                process.wait()
                raise

    def _read_log_tail(self, log_path: Path, lines: int = 100) -> str:
        """Read last N lines of log file efficiently."""
        if not log_path.exists():
            return ""

        # Simple implementation using deque for tail
        from collections import deque
        with open(log_path, 'r') as f:
            return "".join(deque(f, lines))

    def _collect_log_content(self, work_dir: Path) -> str:
        """Helper to collect logs from stdout.log and stderr.log."""
        stdout_path = work_dir / "stdout.log"
        stderr_path = work_dir / "stderr.log"

        log_content_parts = []
        if stdout_path.exists():
            out_tail = self._read_log_tail(stdout_path)
            if out_tail.strip():
                log_content_parts.append(f"--- STDOUT ---\n{out_tail}")

        if stderr_path.exists():
            err_tail = self._read_log_tail(stderr_path)
            if err_tail.strip():
                log_content_parts.append(f"--- STDERR ---\n{err_tail}")

        return "\n".join(log_content_parts)

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

            if parse_result.halt_detected and max_gamma is None:
                max_gamma = 999.9

        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory {traj_path} not found.")

        # Read last frame safely using iread to avoid loading full trajectory into memory
        # This iterates over the file and keeps only the latest frame reference
        atoms = None
        for frame in ase.io.iread(traj_path, format="lammps-dump-text"): # type: ignore[no-untyped-call]
            atoms = frame
            # Ensure we don't hold references to previous frames implicitly
            # (though simple reassignment should handle it in CPython)

        if atoms is None:
             raise ValueError(f"Trajectory {traj_path} is empty or invalid.")

        # Restore symbols
        if len(atoms) == len(original_structure.symbols):
            atoms.set_chemical_symbols(original_structure.symbols)  # type: ignore[no-untyped-call]

        return Structure.from_ase(atoms), traj_path, max_gamma
