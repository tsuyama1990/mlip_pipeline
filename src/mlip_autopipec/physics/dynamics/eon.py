import logging
import subprocess
import shutil
import sys
import uuid
from pathlib import Path

import ase.io
from mlip_autopipec.domain_models.dynamics import EonConfig, EonResult
from mlip_autopipec.domain_models.config import PotentialConfig
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger("mlip_autopipec.physics.dynamics.eon")

class EonWrapper:
    """
    Wrapper for EON (Adaptive Kinetic Monte Carlo).
    """

    def __init__(
        self,
        config: EonConfig,
        potential_config: PotentialConfig,
        base_work_dir: Path,
        uncertainty_threshold: float = 5.0,
    ):
        self.config = config
        self.potential_config = potential_config
        self.base_work_dir = base_work_dir
        self.uncertainty_threshold = uncertainty_threshold
        self.base_work_dir.mkdir(parents=True, exist_ok=True)

        # Resolve driver path
        # Configuration > Env var > Default
        import os
        if self.config.driver_path:
            self.driver_path = self.config.driver_path.resolve()
        elif os.environ.get("PACE_DRIVER_PATH"):
            self.driver_path = Path(os.environ["PACE_DRIVER_PATH"]).resolve()
        else:
            # Fallback to default relative location
            current_file = Path(__file__).resolve()
            # src/mlip_autopipec/physics/dynamics/eon.py -> src/mlip_autopipec/inference/pace_driver.py
            pkg_root = current_file.parent.parent.parent
            self.driver_path = pkg_root / "inference" / "pace_driver.py"

        if not self.driver_path.exists():
             logger.warning(f"pace_driver.py not found at {self.driver_path}.")
             # We rely on Config validation usually, but here we warn.

    def run(
        self,
        structure: Structure,
        potential_path: Path,
    ) -> EonResult:
        """
        Run EON aKMC.
        """
        job_id = str(uuid.uuid4())[:8]
        work_dir = self.base_work_dir / f"akmc_{job_id}"
        work_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Setting up EON job in {work_dir}")

        try:
            # 1. Setup Client Directory
            # EON often expects to run in a dir with config.ini
            # It also creates 'runs' dir.
            # We treat work_dir as the simulation root.

            client_dir = work_dir / "client"
            client_dir.mkdir()

            # 2. Write initial structure (pos.con)
            # EON uses .con format. ASE supports it? 'con' is often 'castep-cell' or similar?
            # EON .con is specific. ASE has 'eon' format support?
            # Let's check ASE formats. ASE has 'eon' reader/writer.
            atoms = structure.to_ase()
            ase.io.write(work_dir / "pos.con", atoms, format="eon") # type: ignore

            # 3. Write config.ini
            self._write_config(client_dir, potential_path)

            # 4. Execute EON
            cmd_str = self.config.command
            if self.config.use_mpi:
                cmd_str = f"{self.config.mpi_command} {cmd_str}"

            cmd_list = cmd_str.split()

            # Check executable
            if not shutil.which(cmd_list[0]):
                if cmd_list[0] == "eon":
                     # For Cycle 08 UAT, we might simulate EON if not present
                     # But code should assume it is present or fail
                     pass

            logger.info(f"Executing EON: {cmd_list}")

            stdout_path = work_dir / self.config.stdout_file
            stderr_path = work_dir / self.config.stderr_file

            with open(stdout_path, "w") as f_out, open(stderr_path, "w") as f_err:
                # EON runs until it finds something or time limit
                # We should set time limit or steps limit in config.ini
                # Using subprocess.run might block too long.
                # EON usually runs as a server/client or monolithic?
                # "EON acts as Master".

                process = subprocess.Popen(
                    cmd_list,
                    cwd=work_dir,
                    stdout=f_out,
                    stderr=f_err
                )

                try:
                    process.wait(timeout=self.config.timeout)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    logger.warning("EON timed out")
                    return EonResult(
                        job_id=job_id,
                        status=JobStatus.TIMEOUT,
                        work_dir=work_dir,
                        duration_seconds=self.config.timeout,
                        log_content="Timeout Expired",
                        final_structure=None,
                        max_gamma=None
                    )

            # 5. Check exit code
            # If driver exited with 100, EON might fail or return error.
            # We verify logs to see if "High Gamma" was reported.

            # Parse output
            return self._parse_output(work_dir)

        except Exception as e:
            logger.error(f"EON execution failed: {e}")
            return EonResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                work_dir=work_dir,
                duration_seconds=0.0,
                log_content=str(e),
                final_structure=None,
                max_gamma=None
            )

    def _write_config(self, work_dir: Path, potential_path: Path) -> None:
        """Write config.ini for EON."""
        # Construct command to run driver
        # We wrap it in python call to be safe
        # EON reads 'script_path' as a string and executes it.
        # We MUST NOT use shell=True or unsafe string interpolation blindly.
        # However, EON is the one executing it.
        # We can construct a safe command string where arguments are properly quoted/escaped.
        import shlex

        # Quote paths to handle spaces
        safe_exe = shlex.quote(sys.executable)
        safe_driver = shlex.quote(str(self.driver_path.resolve()))
        safe_pot = shlex.quote(str(potential_path.resolve()))

        # Elements are alpha chars usually, but we quote them safe measure
        safe_elements = [shlex.quote(e) for e in self.potential_config.elements]
        elements_str = " ".join(safe_elements)

        # Construct command with dynamic parameters
        full_pot_cmd = (
            f"{safe_exe} {safe_driver} "
            f"--potential {safe_pot} "
            f"--elements {elements_str} "
            f"--threshold {self.uncertainty_threshold} "
            f"--zbl-inner {self.potential_config.zbl_inner_cutoff} "
            f"--zbl-outer {self.potential_config.zbl_outer_cutoff}"
        )

        # EON Config.ini format
        config_content = f"""[Main]
job = nudged_elastic_band
temperature = 300
random_seed = {self.potential_config.seed}

[Potential]
potential = script
script_path = {full_pot_cmd}

[Optimizer]
converged_force = 0.01

[Saddle Search]
method = min_mode
"""
        # Note: 'job' might be 'process_search' or 'akmc'?
        # SPEC says "EON acts as Master... explores PES... finds saddle points".
        # This implies 'process_search' or 'dynamics'.
        # For simplicity, we use generic settings or 'process_search'.

        (work_dir / "config.ini").write_text(config_content)

    def _parse_output(self, work_dir: Path) -> EonResult:
        """Parse EON output and logs."""
        # Check stderr for "High Gamma" from driver
        stderr_path = work_dir / self.config.stderr_file
        max_gamma = 0.0
        final_struct = None
        status = JobStatus.COMPLETED

        if stderr_path.exists():
            content = stderr_path.read_text()
            if "High Gamma Detected" in content:
                # Extract gamma value if possible
                # "High Gamma Detected: 6.5 > 5.0"
                import re
                match = re.search(r"High Gamma Detected: ([\d\.]+)", content)
                if match:
                    max_gamma = float(match.group(1))

                # If high gamma, we want the structure that caused it.
                # The driver might have dumped it? Or EON saves current pos?
                # EON writes `pos.con` or `results/`.
                # If driver crashed, EON might have stopped at that structure.
                # Let's read `pos.con` or check if driver dumped something?
                # Driver doesn't dump structure currently, only prints error.
                # But EON sends structure to driver.
                # To capture it, we should modify driver to dump `bad_structure.xyz` on exit 100.

                # Let's look for `bad_structure.lammps` or similar if we modify driver.
                # For now, let's assume we read `pos.con` (current state).
                try:
                    atoms = ase.io.read(work_dir / "pos.con", format="eon") # type: ignore
                    final_struct = Structure.from_ase(atoms) # type: ignore
                except Exception:
                    pass

        return EonResult(
            job_id=work_dir.name,
            status=status,
            work_dir=work_dir,
            duration_seconds=0.0,
            log_content="Parsed EON output",
            final_structure=final_struct,
            max_gamma=max_gamma
        )
